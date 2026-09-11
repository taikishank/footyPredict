"""Is each league-season in the fixture cache actually complete?

A season missing scattered fixtures is worse than a season left out: rolling
form windows silently span a gap, so a team's "last 5 matches" quietly becomes
its last 5 *recorded* matches, and the label distribution skews with it. Better
to find the holes and drop whole seasons than to train through them.

Three checks, all against the existing cache - no API calls:

1. **Season size** - does the league-season have the fixture count a complete
   double round-robin implies for its team count?
2. **Per-team** - is any team short of the 2*(teams-1) fixtures it should play?
   A team missing 1-8 fixtures is the dangerous case: too few to notice, enough
   to distort that team's form features.
3. **Round gaps** - is any matchweek missing or short? Rounds come back as ids
   rather than numbers, so they're ordered by first kickoff and checked for
   having the expected fixture count (teams/2). An uneven round is only a
   problem when fixtures are genuinely absent: postponed matches get reassigned
   to a different round id, which leaves a short round and a long one in a
   season that is in fact whole. So short rounds are reported, but they only
   raise an alert when the season or per-team counts are also short.

Usage:
    python3 ml/player_stats/check_season_completeness.py
    python3 ml/player_stats/check_season_completeness.py --league 8
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
ALL_FIXTURES_DIR = ML_DIR / "data" / "raw" / "all_fixtures"
INDEX_PATH = ML_DIR / "data" / "player_stats_season_index_v2.json"
REPORT_PATH = Path(__file__).resolve().parent / "season_completeness.json"

LEAGUES = {8: "premier_league", 82: "bundesliga", 301: "ligue1",
           384: "serie_a", 564: "laliga"}

# A cached fixture row is not evidence that a match happened: Ligue 1 2019/20
# has all 380 rows, 101 of them Cancelled or Postponed. So completeness has to
# be counted over *played* fixtures, not over rows - otherwise a COVID-
# truncated season reads as complete and every team silently loses ~10 matches
# from its form history. State names resolved via the `state` include
# (/core/states is not on our plan).
PLAYED_STATE_IDS = {5, 7}           # Full Time, After Extra Time
NOT_STARTED_STATE_ID = 1            # scheduled, simply hasn't happened yet
NEVER_PLAYED_STATE_IDS = {10: "postponed", 12: "cancelled",
                          15: "abandoned", 17: "awarded"}


def build_index() -> list[dict]:
    """One record per cached fixture: the season/round/team fields the checks need.

    Scanning 41k files takes a few minutes, so the result is cached.
    """
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    records = []
    for path in ALL_FIXTURES_DIR.glob("*.json"):
        fixture = json.loads(path.read_text())
        records.append({
            "fixture_id": fixture["id"],
            "league_id": fixture.get("league_id"),
            "season_id": fixture.get("season_id"),
            "round_id": fixture.get("round_id"),
            "state_id": fixture.get("state_id"),
            "starting_at": fixture.get("starting_at"),
            "team_ids": [p.get("id") for p in fixture.get("participants") or []],
            # carried so alerts can name the team rather than print a bare id
            "team_names": {str(p.get("id")): p.get("name")
                           for p in fixture.get("participants") or []},
        })
    INDEX_PATH.write_text(json.dumps(records))
    return records


def check_season(fixtures: list[dict]) -> dict:
    """Completeness findings for one league-season."""
    played = [f for f in fixtures if f["state_id"] in PLAYED_STATE_IDS]
    not_started = [f for f in fixtures if f["state_id"] == NOT_STARTED_STATE_ID]
    never_played = Counter(NEVER_PLAYED_STATE_IDS[f["state_id"]] for f in fixtures
                           if f["state_id"] in NEVER_PLAYED_STATE_IDS)

    # team and round counts are over the full card (so the expected size is
    # right) but completeness is judged on what was actually played
    team_counts: Counter = Counter()
    for fixture in fixtures:
        for team_id in fixture["team_ids"]:
            team_counts[team_id] += 1
    played_per_team: Counter = Counter()
    for fixture in played:
        for team_id in fixture["team_ids"]:
            played_per_team[team_id] += 1
    team_count = len(team_counts)
    # a double round-robin: every team plays every other home and away
    expected_per_team = 2 * (team_count - 1) if team_count > 1 else 0
    expected_total = team_count * expected_per_team // 2

    short_teams = {str(team_id): played_per_team.get(team_id, 0)
                   for team_id in sorted(team_counts)
                   if played_per_team.get(team_id, 0) < expected_per_team}

    # rounds carry ids, not numbers, so order them by first kickoff and report
    # positions - a short round is a missing matchweek or part of one
    by_round: dict[int, list[dict]] = defaultdict(list)
    for fixture in played:
        if fixture["round_id"] is not None:
            by_round[fixture["round_id"]].append(fixture)
    ordered = sorted(by_round.items(),
                     key=lambda kv: min(f["starting_at"] or "" for f in kv[1]))
    expected_per_round = team_count // 2 if team_count else 0
    short_rounds = [
        {"position": position, "round_id": round_id,
         "fixtures": len(round_fixtures), "expected": expected_per_round,
         "first_kickoff": min(f["starting_at"] or "" for f in round_fixtures)}
        for position, (round_id, round_fixtures) in enumerate(ordered, start=1)
        if len(round_fixtures) < expected_per_round
    ]

    return {
        "teams": team_count,
        "fixtures": len(fixtures),
        "played_fixtures": len(played),
        "expected_fixtures": expected_total,
        "missing_fixtures": max(expected_total - len(played), 0),
        "not_started_fixtures": len(not_started),
        "never_played": dict(never_played),
        "expected_per_team": expected_per_team,
        "rounds": len(by_round),
        "expected_rounds": expected_per_team,
        "short_teams": short_teams,
        "short_rounds": short_rounds,
        "first_kickoff": min((f["starting_at"] or "" for f in fixtures), default=""),
        "last_kickoff": max((f["starting_at"] or "" for f in fixtures), default=""),
    }


def verdict(finding: dict) -> str:
    """COMPLETE / IN PROGRESS / DROP - what to do with this season.

    The 1-8 band is called out separately because it's the case that does real
    damage quietly: a handful of absent fixtures per team is invisible in
    aggregate counts but silently stretches that team's rolling form window.
    """
    if finding["not_started_fixtures"] == finding["fixtures"]:
        return "FUTURE"
    if finding["not_started_fixtures"] and not finding["never_played"]:
        return "IN PROGRESS"
    missing = finding["missing_fixtures"]
    if not missing and not finding["short_teams"]:
        # short rounds alone mean postponements were rescheduled into other
        # rounds, not that anything is absent - every team still played its full
        # card. Worth showing, not worth dropping a season over.
        return "COMPLETE (uneven rounds)" if finding["short_rounds"] else "COMPLETE"
    worst = max((finding["expected_per_team"] - played
                 for played in finding["short_teams"].values()), default=0)
    return "DROP (1-8 missing per team)" if 1 <= worst <= 8 else "DROP"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", type=int, default=None, help="restrict to one league id")
    args = parser.parse_args()

    records = build_index()
    print(f"indexed {len(records)} cached fixtures")
    team_names = {}
    for record in records:
        team_names.update(record.get("team_names") or {})

    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        if record["league_id"] not in LEAGUES:
            continue
        if args.league and record["league_id"] != args.league:
            continue
        grouped[(record["league_id"], record["season_id"])].append(record)

    report = {}
    alerts = []
    uneven = []
    print(f"\n{'league':<16}{'season':>8}{'teams':>6}{'played':>8}{'exp':>6}"
          f"{'rounds':>8}{'short teams':>12}  verdict")
    for (league_id, season_id), fixtures in sorted(
            grouped.items(), key=lambda kv: min(f["starting_at"] or "" for f in kv[1])):
        finding = check_season(fixtures)
        finding["verdict"] = v = verdict(finding)
        report[f"{LEAGUES[league_id]}/{season_id}"] = finding
        print(f"{LEAGUES[league_id]:<16}{season_id:>8}{finding['teams']:>6}"
              f"{finding['played_fixtures']:>8}{finding['expected_fixtures']:>6}"
              f"{finding['rounds']:>8}{len(finding['short_teams']):>12}  {v}")
        if v.startswith("DROP"):
            alerts.append((LEAGUES[league_id], season_id, finding))
        elif finding["short_rounds"] and v.startswith("COMPLETE"):
            uneven.append((LEAGUES[league_id], season_id, finding))

    if alerts:
        print(f"\n{'=' * 70}\nALERTS: {len(alerts)} season(s) with holes\n{'=' * 70}")
        for league, season_id, finding in alerts:
            print(f"\n{league} season {season_id} "
                  f"({finding['first_kickoff'][:10]} to {finding['last_kickoff'][:10]})")
            print(f"  {finding['played_fixtures']} played of {finding['expected_fixtures']} "
                  f"expected ({finding['missing_fixtures']} missing)")
            if finding["never_played"]:
                print("  never played: " + ", ".join(
                    f"{count} {reason}" for reason, count in sorted(finding["never_played"].items())))
            if finding["not_started_fixtures"]:
                print(f"  {finding['not_started_fixtures']} still scheduled (not started)")
            if finding["short_teams"]:
                print(f"  teams short of {finding['expected_per_team']} fixtures:")
                for team_id, played in finding["short_teams"].items():
                    name = team_names.get(team_id, f"team {team_id}")
                    print(f"    {name:<24} played {played} "
                          f"({finding['expected_per_team'] - played} missing)")
            for short in finding["short_rounds"]:
                print(f"  round position {short['position']} (id {short['round_id']}, "
                      f"from {short['first_kickoff'][:10]}): {short['fixtures']} fixtures, "
                      f"expected {short['expected']}")
    else:
        print("\nno holes found - every completed season is whole "
              "(0 missing fixtures, 0 short teams)")

    if uneven:
        print(f"\n{len(uneven)} complete season(s) have unevenly-sized rounds "
              f"(postponements rescheduled into other rounds, nothing missing):")
        for league, season_id, finding in uneven:
            positions = ", ".join(str(s["position"]) for s in finding["short_rounds"][:8])
            more = "..." if len(finding["short_rounds"]) > 8 else ""
            print(f"  {league:<16} season {season_id:<6} "
                  f"{len(finding['short_rounds'])} short round(s) at position {positions}{more}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
