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
REQUEST_DELAY_SECONDS = 1.0  # shared account quota with production app - stay polite
PER_PAGE = 50


def fetch_all_fixture_ids(session: requests.Session) -> list[int]:
    # page-based pagination caps out at offset 20000 ("filtering and
    # complexity exceptions") - cursor-based pagination has no such limit.
    # next_cursor comes back as a full URL (cursor param already baked in),
    # not a bare token, so we follow it directly rather than re-wrapping it.
    fixture_ids = []
    url = f"{BASE_URL}/fixtures"
    params = {"per_page": PER_PAGE}
    page_num = 1

    while True:
        print(f"fetching fixtures page {page_num}...")
        response = session.get(url, params=params)
        response.raise_for_status()
        payload = response.json()

        fixture_ids.extend(fixture["id"] for fixture in payload["data"])

        pagination = payload["pagination"]
        if not pagination["has_more"]:
            break

        url = pagination["next_cursor"]
        params = {}
        page_num += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    return fixture_ids


def main() -> None:
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}

    fixture_ids = fetch_all_fixture_ids(session)
    print(f"fetched {len(fixture_ids)} fixture ids")

    FIXTURE_IDS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIXTURE_IDS_DIR / "all_fixtures.json"
    output_path.write_text(json.dumps(fixture_ids))


if __name__ == "__main__":
    main()
