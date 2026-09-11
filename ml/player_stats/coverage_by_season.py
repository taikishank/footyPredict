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


def sample_per_year(index: dict[str, list], per_year: int) -> dict[int, list[int]]:
    by_year: dict[int, list[int]] = defaultdict(list)
    for fixture_id, (starting_at, _) in sorted(index.items(), key=lambda kv: kv[1][0]):
        year = int(starting_at[:4])
        if len(by_year[year]) < per_year:
            by_year[year].append(int(fixture_id))
    return dict(sorted(by_year.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-year", type=int, default=4)
    args = parser.parse_args()

    index = build_index()
    print(f"indexed {len(index)} cached fixtures")
    samples = sample_per_year(index, args.per_year)

    session = requests.Session()
    session.params = {"api_token": os.environ["SPORTMONKS_API_KEY"]}

    report = {}
    print(f"\n{'year':>6} {'fixtures':>8} {'with XI':>8} {'players':>8} {'stat types':>11} {'stats/player':>13}")
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
        report[year] = {"sampled": len(fixture_ids), "with_lineup": lineups,
                        "players": players, "distinct_stat_types": len(stat_types),
                        "stats_per_player": round(per_player, 1)}
        print(f"{year:>6} {len(fixture_ids):>8} {lineups:>8} {players:>8} "
              f"{len(stat_types):>11} {per_player:>13.1f}")

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {REPORT_PATH}")


if __name__ == "__main__":
    main()
