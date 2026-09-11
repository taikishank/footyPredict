"""How far back do per-player stats actually go?

The smoke test turned up something the availability probe didn't: a 2010
fixture returns a full 36-man team sheet but only ~18 per-player stat rows,
where a 2026 fixture returns ~60 stat types for each of 40 players. Lineups
being *present* historically is not the same as them being *useful*, and the
difference decides how much training data the Phase 9 challenger really has.

This samples fixtures per season-year and reports coverage, so the Phase 9
train/validation split can start where the data is actually dense instead of
where the fixture cache happens to begin.

Usage:
    python3 ml/player_stats/coverage_by_season.py --per-year 4
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import load_dotenv

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_lineups as pl  # noqa: E402
from fetch_all_fixtures_data import compute_sleep_seconds  # noqa: E402
from fetch_lineups import INCLUDE, BASE_URL  # noqa: E402

load_dotenv()

ALL_FIXTURES_DIR = ML_DIR / "data" / "raw" / "all_fixtures"
INDEX_PATH = ML_DIR / "data" / "player_stats_fixture_index.json"
REPORT_PATH = Path(__file__).resolve().parent / "coverage_by_season.json"


def build_index() -> dict[str, list]:
    """fixture_id -> (starting_at, league_id), cached.

    Scanning 41k JSON files takes a couple of minutes, and Phase 9's
    chronological split needs the same mapping, so it's worth keeping.
    """
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    index = {}
    for path in ALL_FIXTURES_DIR.glob("*.json"):
        fixture = json.loads(path.read_text())
        if fixture.get("starting_at"):
            index[str(fixture["id"])] = [fixture["starting_at"], fixture.get("league_id")]
    INDEX_PATH.write_text(json.dumps(index))
    return index


def sample_per_year(index: dict[str, list], per_year: int,
                    years: list[int] | None = None,
                    by_league: bool = False) -> dict[object, list[int]]:
    """Spread the sample evenly across each year rather than taking the first N.

    Taking the earliest N fixtures of a year samples one round of one or two
    competitions, which is exactly the wrong shape for asking "is coverage thin
    in this era" - a single league's reporting gap would read as a league-wide
    cliff. Evenly-spaced picks (optionally split per league) make the answer
    trustworthy enough to set a training window on.
    """
    grouped: dict[object, list[tuple[str, int]]] = defaultdict(list)
    for fixture_id, (starting_at, league_id) in index.items():
        year = int(starting_at[:4])
        if years and year not in years:
            continue
        key = (year, league_id) if by_league else year
        grouped[key].append((starting_at, int(fixture_id)))

    sampled: dict[object, list[int]] = {}
    for key, entries in grouped.items():
        entries.sort()
        if len(entries) <= per_year:
            picks = entries
        else:
            step = len(entries) / per_year
            picks = [entries[int(i * step)] for i in range(per_year)]
        sampled[key] = [fixture_id for _, fixture_id in picks]
    return dict(sorted(sampled.items(), key=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-year", type=int, default=4,
                        help="fixtures to sample per group, spread across the year")
    parser.add_argument("--years", type=str, default=None,
                        help="comma-separated years to restrict to, e.g. 2014,2015,2016")
    parser.add_argument("--by-league", action="store_true",
                        help="group by (year, league) to catch a single league skewing a year")
    parser.add_argument("--out", type=str, default=None, help="where to write the JSON report")
    args = parser.parse_args()
    years = [int(y) for y in args.years.split(",")] if args.years else None

    index = build_index()
    print(f"indexed {len(index)} cached fixtures")
    samples = sample_per_year(index, args.per_year, years=years, by_league=args.by_league)

    session = requests.Session()
    session.params = {"api_token": os.environ["SPORTMONKS_API_KEY"]}

    report = {}
    label = "year/league" if args.by_league else "year"
    print(f"\n{label:>12} {'fixtures':>8} {'with XI':>8} {'players':>8} {'stat types':>11} {'stats/player':>13}")
    for year, fixture_ids in samples.items():
        lineups = players = stat_rows = 0
        stat_types: set[int] = set()
        for fixture_id in fixture_ids:
            response = session.get(f"{BASE_URL}/fixtures/{fixture_id}", params={"include": INCLUDE})
            if response.status_code != 200:
                continue
            payload = response.json()
            fixture = payload["data"]
            if pl.lineup_source(fixture):
                lineups += 1
            players += len(pl.parse_lineups(fixture))
            rows = pl.parse_player_stats(fixture)
            stat_rows += len(rows)
            stat_types.update(row["type_id"] for row in rows if row["type_id"])
            rate = payload["rate_limit"]
            time.sleep(compute_sleep_seconds(rate["remaining"], rate["resets_in_seconds"]))
        per_player = stat_rows / players if players else 0
        report[str(year)] = {"sampled": len(fixture_ids), "with_lineup": lineups,
                        "players": players, "distinct_stat_types": len(stat_types),
                        "stats_per_player": round(per_player, 1)}
        print(f"{str(year):>12} {len(fixture_ids):>8} {lineups:>8} {players:>8} "
              f"{len(stat_types):>11} {per_player:>13.1f}")

    out_path = Path(args.out) if args.out else REPORT_PATH
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
