"""The same backfill as fetch_lineups.py, 50 fixtures per request instead of 1.

`/fixtures/between/{from}/{to}` accepts the same includes as the single-fixture
endpoint and returns up to 50 fixtures per call, each with its full
`lineups.details`, `lineups.player` and `sidelined.sideline` payload - verified
against a single-fixture control, which returns no stat type this misses. That
turns ~41,800 requests into ~840, and a ~14h job into well under an hour.

Nothing is traded away for the speed: it writes the identical per-fixture gzip
files to the identical cache, so the two fetchers are interchangeable and
`parse_lineups.py` neither knows nor cares which produced a file. Use
fetch_lineups.py for a handful of specific fixtures (the live poller's case) and
this for bulk history.

Resumable at month granularity: a month whose fixtures are all cached is skipped
without a request.

Usage:
    python3 ml/player_stats/fetch_lineups_bulk.py
    python3 ml/player_stats/fetch_lineups_bulk.py --from 2016-01 --to 2016-12
"""

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_lineups import (  # noqa: E402
    BASE_URL,
    CACHE_DIR,
    FIXTURE_IDS_PATH,
    INCLUDE,
    SPORTMONKS_API_KEY,
    cache_path,
    sleep_seconds_for,
    write_cached,
)

load_dotenv()

ML_DIR = Path(__file__).resolve().parent.parent
SEASON_INDEX_PATH = ML_DIR / "data" / "player_stats_season_index_v2.json"

LEAGUE_FILTER = "fixtureLeagues:8,82,301,384,564"
# 50 is the ceiling in practice: per_page=100 comes back capped at 25 by payload
# size, so asking for more just costs an extra round trip.
PER_PAGE = 50
MAX_PAGES_PER_MONTH = 40  # a month of 5 leagues is ~150 fixtures; this is slack


def months(start: date, end: date) -> list[tuple[date, date]]:
    """Inclusive [first, last] day pairs, one per calendar month."""
    spans = []
    cursor = start.replace(day=1)
    while cursor <= end:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        spans.append((cursor, next_month - timedelta(days=1)))
        cursor = next_month
    return spans


def fixture_ids_by_month() -> dict[str, set[int]]:
    """Which fixtures the cache expects per month, so a done month can be skipped."""
    by_month: dict[str, set[int]] = {}
    for record in json.loads(SEASON_INDEX_PATH.read_text()):
        starting_at = record.get("starting_at")
        if starting_at:
            by_month.setdefault(starting_at[:7], set()).add(record["fixture_id"])
    return by_month


def fetch_month(session: requests.Session, first: date, last: date,
                reserve: int) -> tuple[int, int]:
    """Page through one month, writing each fixture. Returns (written, seen)."""
    written = seen = 0
    page = 1
    while page <= MAX_PAGES_PER_MONTH:
        response = session.get(
            f"{BASE_URL}/fixtures/between/{first}/{last}",
            params={"include": INCLUDE, "per_page": PER_PAGE, "page": page,
                    "filters": LEAGUE_FILTER},
        )
        if response.status_code == 429:
            print("    hit 429, backing off 60s...")
            time.sleep(60)
            continue
        response.raise_for_status()
        payload = response.json()

        for fixture in payload.get("data") or []:
            seen += 1
            if not cache_path(fixture["id"]).exists():
                write_cached(fixture["id"], fixture)
                written += 1

        rate_limit = payload.get("rate_limit") or {}
        if rate_limit:
            time.sleep(sleep_seconds_for(rate_limit.get("remaining", 1),
                                         rate_limit.get("resets_in_seconds", 60), reserve))
        if not (payload.get("pagination") or {}).get("has_more"):
            break
        page += 1
    return written, seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM (inclusive)")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM (inclusive)")
    parser.add_argument("--reserve", type=int, default=0,
                        help="requests/hr to leave for production (default 0: this is "
                             "~840 requests total, so it barely touches the quota)")
    args = parser.parse_args()

    expected = fixture_ids_by_month()
    keys = sorted(expected)
    start = date.fromisoformat(f"{args.date_from or keys[0]}-01")
    end_key = args.date_to or keys[-1]
    end = date.fromisoformat(f"{end_key}-01")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    total_ids = len(json.loads(FIXTURE_IDS_PATH.read_text()))
    already = len(list(CACHE_DIR.glob("*.json.gz")))
    print(f"{already}/{total_ids} fixtures already cached")

    spans = months(start, end)
    print(f"sweeping {len(spans)} month(s) from {start:%Y-%m} to {end:%Y-%m}, "
          f"{PER_PAGE} fixtures per request")

    written = skipped_months = 0
    t0 = time.time()
    for i, (first, last) in enumerate(spans, start=1):
        month_key = f"{first:%Y-%m}"
        wanted = expected.get(month_key, set())
        if wanted and all(cache_path(fid).exists() for fid in wanted):
            skipped_months += 1
            continue
        month_written, seen = fetch_month(session, first, last, args.reserve)
        written += month_written
        if i % 12 == 0 or month_written:
            elapsed = time.time() - t0
            print(f"[{i}/{len(spans)}] {month_key}: +{month_written} of {seen} seen "
                  f"({written} written this run, {elapsed / 60:.1f} min elapsed)")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {written} fixtures written, "
          f"{skipped_months} month(s) already complete")
    print(f"cache now holds {len(list(CACHE_DIR.glob('*.json.gz')))}/{total_ids} fixtures")


if __name__ == "__main__":
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}
    main()
