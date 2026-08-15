import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SPORTMONKS_API_KEY = os.environ["SPORTMONKS_API_KEY"]
BASE_URL = "https://api.sportmonks.com/v3/football"
FIXTURE_IDS_DIR = Path(__file__).parent / "data" / "fixture_ids"
RAW_DIR = Path(__file__).parent / "data" / "raw" / "all_fixtures"
FAILED_IDS_PATH = Path(__file__).parent / "data" / "failed_fixture_ids.json"

# account quota (3000/hr) is shared with a production app - keep this much
# headroom free for it at all times, even mid-hour.
RESERVED_FOR_PROD = 25
FALLBACK_DELAY_SECONDS = 1.25  # used only until the first response tells us remaining/reset


def compute_sleep_seconds(remaining: int, resets_in_seconds: int) -> float:
    """Given the account's current rate-limit state (as of the last response),
    return how long to sleep before firing the next request.

    Spends only the budget above RESERVED_FOR_PROD, spread evenly across the
    time left until reset - so a quiet prod app lets us go faster, while a
    busy one (remaining approaching RESERVED_FOR_PROD) makes us back off
    instead of racing to grab the last available requests.
    """
    usable = remaining - RESERVED_FOR_PROD
    if usable <= 0:
        # already at (or past) the reserved floor - wait out the reset
        # entirely rather than risk taking a request prod needs.
        return resets_in_seconds + 1
    return max(resets_in_seconds / usable, 0.05)


MAX_TRANSIENT_RETRIES = 3
TRANSIENT_STATUS_CODES = {502, 503, 504}


def fetch_fixture(session: requests.Session, fixture_id: int, retries: int = 0) -> tuple[dict, dict] | None:
    url = f"{BASE_URL}/fixtures/{fixture_id}"
    response = session.get(url, params={"include": "scores;participants;statistics"})
    if response.status_code == 429:
        # shared account got rate-limited from under us - back off a full
        # reset window and let the caller retry.
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


def record_failed_fixture(fixture_id: int) -> None:
    failed_ids = set(json.loads(FAILED_IDS_PATH.read_text())) if FAILED_IDS_PATH.exists() else set()
    failed_ids.add(fixture_id)
    FAILED_IDS_PATH.write_text(json.dumps(sorted(failed_ids), indent=2))


def load_all_fixture_ids() -> list[int]:
    path = FIXTURE_IDS_DIR / "all_fixtures.json"
    return json.loads(path.read_text())


def fetch_all(session: requests.Session) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fixture_ids = load_all_fixture_ids()

    remaining_ids = [fid for fid in fixture_ids if not (RAW_DIR / f"{fid}.json").exists()]
    print(f"{len(fixture_ids) - len(remaining_ids)} already cached, {len(remaining_ids)} to fetch")

    sleep_seconds = FALLBACK_DELAY_SECONDS
    for i, fixture_id in enumerate(remaining_ids, start=1):
        result = fetch_fixture(session, fixture_id)
        if result is None:
            continue
        fixture, rate_limit = result

        cache_path = RAW_DIR / f"{fixture_id}.json"
        cache_path.write_text(json.dumps(fixture, indent=2))

        if i % 100 == 0:
            print(f"[{i}/{len(remaining_ids)}] fixture {fixture_id} "
                  f"(remaining={rate_limit['remaining']}, resets_in={rate_limit['resets_in_seconds']}s)")

        sleep_seconds = compute_sleep_seconds(rate_limit["remaining"], rate_limit["resets_in_seconds"])
        time.sleep(sleep_seconds)


def main() -> None:
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}
    fetch_all(session)


if __name__ == "__main__":
    main()
