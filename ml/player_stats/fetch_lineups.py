"""Phase 8 backfill: lineups + per-player stats for every cached fixture.

Writes to its own cache, `data/raw/player_lineups/`, and never touches
`data/raw/all_fixtures/` - the existing model's training input stays
byte-identical, so the challenger can be built and compared without any risk
to the incumbent.

Two decisions worth knowing before you run this (both measured, see
FINDINGS_DATA_AVAILABILITY.md):

- **No stat-type filter.** The payload carries ~60 per-player stat types and we
  keep all of them. Trimming is free at feature-engineering time; recovering a
  type we chose not to fetch means re-running this whole job.
- **gzipped, compact JSON.** ~9 KiB/fixture instead of 242 KiB pretty-printed,
  which is what makes keeping all 60 types affordable: ~0.4 GiB for the full
  backfill rather than ~9.7 GiB.

Cost: one request per fixture, ~41.8k fixtures, ~14h at the rate limiter's
pace. Resumable - already-cached fixtures are skipped, so Ctrl+C is safe.

Usage:
    python3 ml/player_stats/fetch_lineups.py              # full backfill
    python3 ml/player_stats/fetch_lineups.py --limit 20   # smoke test
"""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ML_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_DIR))

# Reuse the existing limiter rather than re-deriving it: it already encodes how
# much of the shared account quota production needs left alone, which is the
# part that's costly to get wrong.
from fetch_all_fixtures_data import (  # noqa: E402
    FALLBACK_DELAY_SECONDS,
    MAX_TRANSIENT_RETRIES,
    RESERVED_FOR_PROD,
    TRANSIENT_STATUS_CODES,
    compute_sleep_seconds,
)

load_dotenv()

SPORTMONKS_API_KEY = os.environ["SPORTMONKS_API_KEY"]
BASE_URL = "https://api.sportmonks.com/v3/football"
FIXTURE_IDS_PATH = ML_DIR / "data" / "fixture_ids" / "all_fixtures.json"
CACHE_DIR = ML_DIR / "data" / "raw" / "player_lineups"
FAILED_IDS_PATH = ML_DIR / "data" / "player_lineups_failed_ids.json"

# lineups.details = per-player stats (the whole point); lineups.player = player
# identity for the players table; sidelined.sideline = the nested object
# carrying category/dates - the bare `sidelined` include returns ids only.
INCLUDE = "lineups.details;lineups.player;sidelined.sideline"


def cache_path(fixture_id: int) -> Path:
    return CACHE_DIR / f"{fixture_id}.json.gz"


def write_cached(fixture_id: int, fixture: dict) -> None:
    # compact separators + gzip: see module docstring. mtime=0 keeps the output
    # byte-identical across runs, so a re-fetch doesn't churn the cache.
    blob = json.dumps(fixture, separators=(",", ":")).encode()
    with gzip.GzipFile(cache_path(fixture_id), "wb", mtime=0) as handle:
        handle.write(blob)


def read_cached(fixture_id: int) -> dict:
    """Read one cached fixture back - the entry point for offline parsing."""
    with gzip.open(cache_path(fixture_id), "rt") as handle:
        return json.load(handle)


def record_failed_fixture(fixture_id: int) -> None:
    failed = set(json.loads(FAILED_IDS_PATH.read_text())) if FAILED_IDS_PATH.exists() else set()
    failed.add(fixture_id)
    FAILED_IDS_PATH.write_text(json.dumps(sorted(failed), indent=2))


def fetch_fixture(session: requests.Session, fixture_id: int, retries: int = 0) -> tuple[dict, dict] | None:
    """Same retry/backoff contract as fetch_all_fixtures_data.fetch_fixture,
    differing only in the includes requested."""
    response = session.get(f"{BASE_URL}/fixtures/{fixture_id}", params={"include": INCLUDE})
    if response.status_code == 429:
        print("hit 429, backing off 60s...")
        time.sleep(60)
        return fetch_fixture(session, fixture_id, retries)
    if response.status_code in TRANSIENT_STATUS_CODES:
        if retries >= MAX_TRANSIENT_RETRIES:
            print(f"fixture {fixture_id} still {response.status_code} after "
                  f"{MAX_TRANSIENT_RETRIES} retries, logging as failed and skipping")
            record_failed_fixture(fixture_id)
            return None
        print(f"fixture {fixture_id} got {response.status_code}, retrying in 5s "
              f"({retries + 1}/{MAX_TRANSIENT_RETRIES})...")
        time.sleep(5)
        return fetch_fixture(session, fixture_id, retries + 1)
    response.raise_for_status()
    payload = response.json()
    return payload["data"], payload["rate_limit"]


def fetch_all(session: requests.Session, limit: int | None = None) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fixture_ids = json.loads(FIXTURE_IDS_PATH.read_text())

    remaining = [fid for fid in fixture_ids if not cache_path(fid).exists()]
    cached = len(fixture_ids) - len(remaining)
    if limit is not None:
        remaining = remaining[:limit]
    print(f"{cached}/{len(fixture_ids)} already cached, {len(remaining)} to fetch")

    sleep_seconds = FALLBACK_DELAY_SECONDS
    with_lineups = 0
    for i, fixture_id in enumerate(remaining, start=1):
        result = fetch_fixture(session, fixture_id)
        if result is None:
            continue
        fixture, rate_limit = result
        write_cached(fixture_id, fixture)
        if fixture.get("lineups"):
            with_lineups += 1

        if i % 100 == 0 or i == len(remaining):
            print(f"[{i}/{len(remaining)}] fixture {fixture_id} "
                  f"({with_lineups}/{i} had lineups, remaining={rate_limit['remaining']}, "
                  f"resets_in={rate_limit['resets_in_seconds']}s, sleep={sleep_seconds:.2f}s)")

        sleep_seconds = compute_sleep_seconds(rate_limit["remaining"], rate_limit["resets_in_seconds"])
        time.sleep(sleep_seconds)

    print(f"done: {with_lineups}/{len(remaining)} fetched fixtures had a lineup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="fetch at most this many uncached fixtures (for smoke tests)")
    args = parser.parse_args()

    print(f"reserving {RESERVED_FOR_PROD} req/hr for production, caching to {CACHE_DIR}")
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}
    fetch_all(session, limit=args.limit)


if __name__ == "__main__":
    main()
