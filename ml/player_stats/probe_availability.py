"""Phase 8, step 0: does our SportMonks plan actually give us player data?

PLAYER_STATS_SPEC.md 9.1-9.3 say to confirm API coverage *before* committing
engineering time. This script answers that with read-only calls and writes
every raw payload it sees to data/raw/player_stats_probe/ so the rest of
Phase 8 (table design, parser) can be built offline against real shapes.

Deliberately standalone: it imports nothing from the existing feature/training
code and writes nowhere that code reads.

Usage:
    python3 ml/player_stats/probe_availability.py
"""

import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SPORTMONKS_API_KEY = os.environ["SPORTMONKS_API_KEY"]
BASE_URL = "https://api.sportmonks.com/v3/football"
# the /my/ endpoints describe the account itself and sit outside /football
ACCOUNT_URL = "https://api.sportmonks.com/v3/my"
CORE_URL = "https://api.sportmonks.com/v3/core"
ML_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ML_DIR / "data" / "raw"
CACHED_FIXTURES_DIR = RAW_DIR / "all_fixtures"
PROBE_DIR = ML_DIR / "data" / "raw" / "player_stats_probe"

# the five leagues the current model trains on (ml/data/raw/<league>/)
LEAGUES = {
    8: "premier_league",
    82: "bundesliga",
    301: "ligue1",
    384: "serie_a",
    564: "laliga",
}

# spec 2: the includes whose availability decides whether this project is viable
FIXTURE_INCLUDE_PROBES = [
    "lineups",
    "lineups.player",
    "lineups.details",
    "sidelined",
    "statistics",  # control: we already know this one works
]

# keep well clear of the prod app's share of the hourly quota (same rationale
# as RESERVED_FOR_PROD in ml/fetch_all_fixtures_data.py). This probe is a few
# dozen requests at most, so a flat, generous delay is enough.
DELAY_SECONDS = 1.5


def get(session: requests.Session, url: str, params: dict | None = None) -> tuple[int, dict]:
    """GET that never raises: we care about 403/404 as *answers*, not errors."""
    response = session.get(url, params=params or {})
    time.sleep(DELAY_SECONDS)
    try:
        body = response.json()
    except ValueError:
        body = {"_non_json_body": response.text[:500]}
    return response.status_code, body


def save(name: str, payload: object) -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    (PROBE_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2))


# --- Q1: what does the plan include? ----------------------------------------

def probe_subscription(session: requests.Session) -> dict:
    """SportMonks exposes the plan's own resource/include allowlist under /my/.

    Asking the account directly is cheaper and less ambiguous than inferring
    "not on our plan" from an empty include on one fixture.
    """
    findings = {}
    for endpoint in ("enrichments", "resources", "leagues", "usage"):
        status, body = get(session, f"{ACCOUNT_URL}/{endpoint}")
        save(f"my_{endpoint}", body)
        rows = body.get("data")
        count = len(rows) if isinstance(rows, list) else 0
        findings[endpoint] = status
        print(f"  /my/{endpoint} -> HTTP {status}, rows={count}")
    return findings


def summarise_enrichments() -> None:
    """Pull the lineup/sideline-relevant rows out of the enrichments payload."""
    path = PROBE_DIR / "my_enrichments.json"
    if not path.exists():
        return
    body = json.loads(path.read_text())
    data = body.get("data")
    if not isinstance(data, (list, dict)):
        return
    interesting = ("lineup", "sideline", "squad", "player", "statistic", "injur", "formation")
    rows = data if isinstance(data, list) else [data]
    granted = [row.get("name") or row.get("description") or "?"
               for row in rows
               if any(word in json.dumps(row).lower() for word in interesting)]
    for name in granted:
        print(f"    granted: {name}")
    # an enrichment simply absent from this list is the plan saying no - spec
    # 9.1's question is answered by what is *missing* as much as what is here.
    for word in ("sideline", "squad", "injur"):
        if not any(word in name.lower() for name in granted):
            print(f"    NOT granted: nothing matching '{word}'")


# --- Q1 (empirical): do the includes come back with real data? --------------

def recent_finished_fixture_per_league() -> dict[int, tuple[int, str]]:
    """Pick the most recent already-played fixture per league from the cache.

    Reusing the cache costs zero extra requests to find test fixtures, and a
    *finished* fixture is the right control: if lineups are on the plan at
    all, a played match is guaranteed to have one. Scanning the per-league
    season directories rather than all_fixtures/ (41k files) keeps this fast.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    best: dict[int, tuple[str, int]] = {}
    for league_id, league in LEAGUES.items():
        for path in (RAW_DIR / league).glob("*.json"):
            fixture = json.loads(path.read_text())
            starting_at = fixture.get("starting_at") or ""
            if not starting_at or starting_at >= now:
                continue
            if league_id not in best or starting_at > best[league_id][0]:
                best[league_id] = (starting_at, fixture["id"])
    return {lid: (fid, when) for lid, (when, fid) in best.items()}


def probe_fixture_includes(session: requests.Session, league_id: int, fixture_id: int,
                          played_at: str = "") -> dict:
    """Request each include on its own so one gated include can't mask another.

    A combined request that 403s tells us nothing about *which* include was
    the problem; one-at-a-time costs a handful more requests and gives a
    per-include verdict.
    """
    league = LEAGUES[league_id]
    print(f"  {league} (fixture {fixture_id}, played {played_at}):")
    results = {}
    for include in FIXTURE_INCLUDE_PROBES:
        status, body = get(session, f"{BASE_URL}/fixtures/{fixture_id}", {"include": include})
        save(f"fixture_{fixture_id}_{include.replace('.', '_')}", body)
        data = body.get("data") or {}
        key = include.split(".")[0]
        value = data.get(key)
        count = len(value) if isinstance(value, list) else (1 if value else 0)
        verdict = "OK" if status == 200 and count else ("EMPTY" if status == 200 else f"HTTP {status}")
        note = body.get("message", "") if status != 200 else ""
        print(f"    include={include:<18} {verdict:<9} rows={count} {note}")
        results[include] = {"status": status, "rows": count, "message": note}
    return results


def load_type_names(session: requests.Session) -> dict[int, str]:
    """type_id -> human name, cached to disk.

    The whole point of spec 3's player_fixture_stats is a type_id -> column
    mapping (the player-level twin of STAT_IDS in feature_lib.py), so the
    names have to be resolved from the API rather than guessed.
    """
    cache = PROBE_DIR / "core_types.json"
    if cache.exists():
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}
    names: dict[int, str] = {}
    page = 1
    while page <= 20:
        status, body = get(session, f"{CORE_URL}/types", {"per_page": 500, "page": page})
        if status != 200:
            break
        for row in body.get("data") or []:
            names[row["id"]] = row["name"]
        if not (body.get("pagination") or {}).get("has_more"):
            break
        page += 1
    save("core_types", {str(k): v for k, v in sorted(names.items())})
    return names


def describe_lineup_shape(session: requests.Session, fixture_id: int) -> None:
    """Print the fields and stat types we'd actually be parsing, per spec 3."""
    path = PROBE_DIR / f"fixture_{fixture_id}_lineups_details.json"
    if not path.exists():
        return
    lineups = (json.loads(path.read_text()).get("data") or {}).get("lineups") or []
    if not lineups:
        print("    (no lineup rows to describe)")
        return
    print(f"    lineup row keys: {sorted(lineups[0].keys())}")
    detail_type_ids = Counter()
    for entry in lineups:
        for detail in entry.get("details") or []:
            detail_type_ids[detail.get("type_id")] += 1
    if not detail_type_ids:
        print("    WARNING: lineups returned but no per-player details -> "
              "player_fixture_stats would be unpopulatable")
        return

    names = load_type_names(session)
    unresolved = [t for t in detail_type_ids if t not in names]
    for type_id in unresolved:
        status, body = get(session, f"{CORE_URL}/types/{type_id}")
        if status == 200:
            names[type_id] = (body.get("data") or {}).get("name", "?")
    if unresolved:
        save("core_types", {str(k): v for k, v in sorted(names.items())})

    print(f"    {len(detail_type_ids)} per-player stat types available:")
    for type_id, count in sorted(detail_type_ids.items()):
        print(f"      {type_id:>6}: {names.get(type_id, 'UNKNOWN'):<32} (on {count} lineup rows)")

    # spec 3 names exactly these columns - confirm each has a real type behind it
    wanted = {
        "minutes_played": "Minutes Played", "rating": "Rating", "goals": "Goals",
        "assists": "Assists", "shots_total": "Shots Total",
        "shots_on_target": "Shots On Target", "key_passes": "Key Passes",
        "tackles": "Tackles", "yellow_cards": "Yellowcards", "red_cards": "Redcards",
    }
    by_name = {v: k for k, v in names.items()}
    print("    spec 3 player_fixture_stats coverage:")
    for column, type_name in wanted.items():
        type_id = by_name.get(type_name)
        present = type_id in detail_type_ids
        status = f"type_id={type_id}" if present else (
            f"type_id={type_id} exists but absent from this fixture" if type_id else "NO MATCHING TYPE")
        print(f"      {column:<16} {'OK ' if present else 'MISSING'} {status}")


def probe_detail_type_filter(session: requests.Session, fixture_id: int,
                             type_ids: tuple[int, ...]) -> None:
    """spec 9.2: what does a lineup backfill actually cost?

    The full lineups.details payload carries ~60 stat types per player. Server-
    side filtering to only the types spec 3 stores is the difference between a
    tractable backfill and a 10GB one, so measure it rather than assume it.
    """
    full = PROBE_DIR / f"fixture_{fixture_id}_lineups_details.json"
    baseline = PROBE_DIR / f"fixture_{fixture_id}_statistics.json"
    filters = "lineupDetailTypes:" + ",".join(str(t) for t in type_ids)
    status, body = get(session, f"{BASE_URL}/fixtures/{fixture_id}",
                       {"include": "lineups.details", "filters": filters})
    save(f"fixture_{fixture_id}_lineups_details_filtered", body)
    sizes = {
        "statistics only (what we cache today)": baseline.stat().st_size if baseline.exists() else 0,
        "lineups.details unfiltered": full.stat().st_size if full.exists() else 0,
        f"lineups.details filtered to {len(type_ids)} types":
            len(json.dumps(body, indent=2).encode()),
    }
    cached = len(list(CACHED_FIXTURES_DIR.glob("*.json")))
    print(f"  payload size per fixture (HTTP {status}):")
    for label, size in sizes.items():
        projected = size * cached / 1024 ** 3
        print(f"    {label:<48} {size / 1024:7.1f} KiB -> {projected:6.2f} GiB over {cached} fixtures")
    print(f"  request cost: one call per fixture either way, so a full re-fetch of all "
          f"{cached} cached fixtures at ~2975 req/hr is ~{cached / 2975:.1f}h of wall clock")


def probe_squad(session: requests.Session, fixture_id: int) -> None:
    """spec 2: /squads/teams/{id} is the 'who's normally available' reference set."""
    path = PROBE_DIR / f"fixture_{fixture_id}_lineups.json"
    if not path.exists():
        return
    fixture = json.loads(path.read_text()).get("data") or {}
    team_ids = sorted({entry.get("team_id") for entry in (fixture.get("lineups") or [])
                       if entry.get("team_id")})[:1]
    for team_id in team_ids:
        status, body = get(session, f"{BASE_URL}/squads/teams/{team_id}", {"include": "player"})
        save(f"squad_team_{team_id}", body)
        data = body.get("data") or []
        print(f"  /squads/teams/{team_id} -> HTTP {status}, players={len(data) if isinstance(data, list) else 0}")


def probe_sidelined(session: requests.Session) -> None:
    """Team-level injury/suspension list - spec 2 wants it for the
    'lineup unknown' window, where it's the only squad signal available."""
    status, body = get(session, f"{BASE_URL}/sidelined")
    save("sidelined_all", body)
    data = body.get("data") or []
    print(f"  /sidelined -> HTTP {status}, rows={len(data) if isinstance(data, list) else 0} "
          f"{body.get('message','')}")
    if isinstance(data, list) and data:
        print(f"    row keys: {sorted(data[0].keys())}")


# --- Q3/Q4: how far ahead are lineups actually confirmed? -------------------

def probe_upcoming_lineup_timing(session: requests.Session, days: int = 3) -> None:
    """For every fixture in the Upcoming tab's window, is a lineup posted yet?

    This is the question that decides whether predicted_lineup() (spec 5) is
    the main path or an edge case: bucket fixtures by hours-to-kickoff and
    see where confirmed lineups start appearing.
    """
    today = datetime.now(timezone.utc)
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=days)).strftime("%Y-%m-%d")
    league_filter = ",".join(str(i) for i in LEAGUES)

    status, body = get(
        session,
        f"{BASE_URL}/fixtures/between/{date_from}/{date_to}",
        {"include": "lineups", "filters": f"fixtureLeagues:{league_filter}", "per_page": 50},
    )
    save("upcoming_with_lineups", body)
    if status != 200:
        print(f"  /fixtures/between -> HTTP {status} {body.get('message','')}")
        return
    fixtures = body.get("data") or []
    # type_id 11 = "Lineup" (starting XI), 12 = "Bench" (per /core/types).
    # A predicted XI comes back as exactly 22 type-11 rows with no bench; a
    # confirmed team sheet carries the bench too. That bench is the only
    # discriminator the payload offers, and spec 5's lineup_source flag
    # depends on it.
    STARTER, BENCH = 11, 12
    buckets: dict[str, Counter] = defaultdict(Counter)
    for fixture in fixtures:
        starting_at = fixture.get("starting_at")
        if not starting_at:
            continue
        kickoff = datetime.strptime(starting_at, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        hours = (kickoff - today).total_seconds() / 3600
        bucket = ("in play/past" if hours < 0 else "<2h" if hours < 2 else
                  "2-6h" if hours < 6 else "6-24h" if hours < 24 else
                  "24-48h" if hours < 48 else ">48h")
        lineups = fixture.get("lineups") or []
        starters = sum(1 for row in lineups if row.get("type_id") == STARTER)
        bench = sum(1 for row in lineups if row.get("type_id") == BENCH)
        kind = "none" if not starters else ("confirmed" if bench else "predicted")
        buckets[bucket][kind] += 1

    print(f"  {len(fixtures)} fixtures in the next {days} days across the 5 leagues")
    print(f"  {'hours to kickoff':<16} {'confirmed':>9} {'predicted':>9} {'no lineup':>9}")
    for bucket in ("in play/past", "<2h", "2-6h", "6-24h", "24-48h", ">48h"):
        counts = buckets.get(bucket)
        if not counts:
            continue
        print(f"  {bucket:<16} {counts['confirmed']:>9} {counts['predicted']:>9} {counts['none']:>9}")


def main() -> None:
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}

    print("\n=== Q1a: plan/subscription metadata ===")
    probe_subscription(session)
    summarise_enrichments()

    print("\n=== Q1b: fixture includes on a finished match per league ===")
    fixtures = recent_finished_fixture_per_league()
    first_fixture_id = None
    for league_id, (fixture_id, played_at) in sorted(fixtures.items()):
        probe_fixture_includes(session, league_id, fixture_id, played_at)
        if first_fixture_id is None:
            first_fixture_id = fixture_id
            describe_lineup_shape(session, first_fixture_id)

    print("\n=== Q2: backfill payload cost ===")
    if first_fixture_id:
        # Minutes Played, Rating, Goals, Assists, Shots Total, Shots On Target,
        # Key Passes, Tackles, Yellowcards - i.e. spec 3's columns.
        probe_detail_type_filter(session, first_fixture_id,
                                 (119, 118, 52, 79, 42, 86, 117, 78, 84))

    print("\n=== Q1c: squads + sidelined endpoints ===")
    if first_fixture_id:
        probe_squad(session, first_fixture_id)
    probe_sidelined(session)

    print("\n=== Q3/Q4: lineup availability across the Upcoming window ===")
    probe_upcoming_lineup_timing(session)

    print(f"\nraw payloads written to {PROBE_DIR}")


if __name__ == "__main__":
    main()
