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
RAW_DIR = Path(__file__).parent / "data" / "raw"
REQUEST_DELAY_SECONDS = 1.0  # shared account quota with production app - stay polite


def load_fixture_ids(league_name: str) -> list[int]:
    path = FIXTURE_IDS_DIR / f"{league_name}.json"
    return json.loads(path.read_text())


def fetch_fixture(session: requests.Session, fixture_id: int) -> dict:
    url = f"{BASE_URL}/fixtures/{fixture_id}"
    response = session.get(url, params={"include": "scores;participants;statistics"})
    response.raise_for_status()
    return response.json()["data"]


def fetch_league_fixtures(session: requests.Session, league_name: str) -> None:
    league_dir = RAW_DIR / league_name
    league_dir.mkdir(parents=True, exist_ok=True)

    fixture_ids = load_fixture_ids(league_name)
    for fixture_id in fixture_ids:
        cache_path = league_dir / f"{fixture_id}.json"
        if cache_path.exists():
            continue

        print(f"fetching {league_name} fixture {fixture_id}...")
        fixture = fetch_fixture(session, fixture_id)
        cache_path.write_text(json.dumps(fixture, indent=2))

        time.sleep(REQUEST_DELAY_SECONDS)


def main() -> None:
    session = requests.Session()
    session.params = {"api_token": SPORTMONKS_API_KEY}

    for path in sorted(FIXTURE_IDS_DIR.glob("*.json")):
        league_name = path.stem
        print(f"processing {league_name}...")
        fetch_league_fixtures(session, league_name)


if __name__ == "__main__":
    main()
