"""Re-fetch cached fixtures that have been played since they were cached.

`fetch_all_fixtures_data.py` skips any fixture already on disk, which is right
for finished matches - they never change - but means a fixture cached while it
was still scheduled stays cached that way forever. 146 fixtures across the five
leagues are currently in that state: kicked off weeks ago, still stored as "Not
Started" with no result and no statistics, so the 2026/27 season to date is
missing from `build_features.py`'s input entirely.

This refreshes exactly those. It reuses `fetch_fixture` from
`fetch_all_fixtures_data`, so the includes, retry/backoff and rate limiting are
identical and the refreshed files are the same shape as the rest of the cache.

Unlike the other fetchers this one *overwrites* files the incumbent model trains
on, so it prints what it would do and changes nothing until you pass --apply.

Usage:
    python3 ml/refresh_stale_fixtures.py                 # dry run
    python3 ml/refresh_stale_fixtures.py --apply
    python3 ml/refresh_stale_fixtures.py --apply --limit 5
"""

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from fetch_all_fixtures_data import (
    FALLBACK_DELAY_SECONDS,
    RAW_DIR,
    compute_sleep_seconds,
    fetch_fixture,
)

load_dotenv()

# A cached fixture in one of these states is settled - whatever happened has
# happened, and re-fetching it would only burn quota. Anything else, once its
# kickoff has passed, is stale. Resolved via the `state` include; /core/states
# is not available on our plan.
SETTLED_STATE_IDS = {
    5: "Full Time",
    7: "After Extra Time",
    8: "Full Time after penalties",
    10: "Postponed",
    12: "Cancelled",
    15: "Abandoned",
    17: "Awarded",
}


def find_stale(now: str) -> list[tuple[int, int, str]]:
    """(fixture_id, cached_state_id, starting_at) for every kicked-off fixture
    whose cached state isn't settled.

    Scans the cache rather than reusing a prebuilt index: this writes to the
    incumbent's training input, so the decision to overwrite should come from
    the files themselves, not from an index that might have drifted.
    """
    stale = []
    for path in RAW_DIR.glob("*.json"):
        fixture = json.loads(path.read_text())
        starting_at = fixture.get("starting_at")
        if not starting_at or starting_at >= now:
            continue
        if fixture.get("state_id") in SETTLED_STATE_IDS:
            continue
        stale.append((fixture["id"], fixture.get("state_id"), starting_at))
    return sorted(stale, key=lambda row: row[2])


def write_cached(fixture_id: int, fixture: dict) -> None:
    """Overwrite atomically and in the cache's existing format.

    A half-written file here would be a corrupt row in the incumbent's training
    data, so the write goes to a temp file and renames over the original.
    """
    path = RAW_DIR / f"{fixture_id}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(fixture, indent=2))
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="actually overwrite the cache (default: dry run)")
    parser.add_argument("--limit", type=int, default=None,
                        help="refresh at most this many fixtures")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"scanning {RAW_DIR} for fixtures kicked off before {now}...")
    stale = find_stale(now)
    if args.limit is not None:
        stale = stale[:args.limit]

    if not stale:
        print("nothing stale - every played fixture in the cache has a settled state")
        return

    print(f"{len(stale)} stale fixture(s); earliest {stale[0][2]}, latest {stale[-1][2]}")
    if not args.apply:
        for fixture_id, state_id, starting_at in stale[:10]:
            print(f"  would refresh {fixture_id} (kicked off {starting_at}, cached state {state_id})")
        if len(stale) > 10:
            print(f"  ... and {len(stale) - 10} more")
        print("\ndry run - nothing written. re-run with --apply to overwrite these files.")
        return

    session = requests.Session()
    session.params = {"api_token": os.environ["SPORTMONKS_API_KEY"]}

    outcomes: Counter = Counter()
    sleep_seconds = FALLBACK_DELAY_SECONDS
    for i, (fixture_id, _, starting_at) in enumerate(stale, start=1):
        result = fetch_fixture(session, fixture_id)
        if result is None:
            outcomes["fetch failed"] += 1
            continue
        fixture, rate_limit = result

        new_state = fixture.get("state_id")
        if new_state in SETTLED_STATE_IDS:
            write_cached(fixture_id, fixture)
            outcomes[SETTLED_STATE_IDS[new_state]] += 1
        else:
            # still unsettled upstream - overwriting would replace one stale row
            # with another, so leave the file alone and say so
            outcomes[f"still unsettled (state {new_state})"] += 1

        if i % 25 == 0 or i == len(stale):
            print(f"[{i}/{len(stale)}] {fixture_id} ({starting_at}) -> state {new_state} "
                  f"(remaining={rate_limit['remaining']})")

        sleep_seconds = compute_sleep_seconds(rate_limit["remaining"], rate_limit["resets_in_seconds"])
        time.sleep(sleep_seconds)

    print("\noutcomes:")
    for outcome, count in outcomes.most_common():
        print(f"  {count:>5}  {outcome}")
    written = sum(count for outcome, count in outcomes.items()
                  if outcome in SETTLED_STATE_IDS.values())
    print(f"\n{written} cache file(s) refreshed - re-run build_features.py to pick them up")


if __name__ == "__main__":
    main()
