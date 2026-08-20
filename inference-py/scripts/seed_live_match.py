"""Seeds a fake fixture (plus the team history build_match_features needs)
and then ticks its live_match_state forward on a timer, so the Angular
dashboard's WebSocket has something to render before ingestor-go's live
poller / SQS / DynamoDB path exists (see PROJECT_SPEC.md Phase 3). All IDs
here are clearly-fake placeholders, not real SportMonks data.

Usage (from inference-py/, with Postgres running via `docker compose up -d`
at the repo root, and inference-py's venv active):

    python scripts/seed_live_match.py
    python scripts/seed_live_match.py --fixture-id 900123 --tick-seconds 2

Then point the Angular dashboard (or a GET to /predictions/{id}/live) at
the printed fixture id while this script is running.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg
from psycopg.types.json import Json

from app.config import POSTGRES_URL

FAKE_HOME_TEAM_ID = 900001
FAKE_AWAY_TEAM_ID = 900002
FAKE_FILLER_OPPONENT_ID = 900099
HISTORY_FIXTURES_PER_TEAM = 5  # matches feature_lib.MIN_WINDOW

# One tuple per tick: (state, home_goals, away_goals). A goal is just a step
# change in the score from the previous tick - main.py's adjust_for_live_state
# reads the diff off of live_match_state, not off any "event" stream.
MATCH_SCRIPT: list[tuple[str, int, int]] = [
    ("NS", 0, 0),
    ("1H", 0, 0),
    ("1H", 0, 0),
    ("1H", 1, 0),  # home goal
    ("1H", 1, 0),
    ("HT", 1, 0),
    ("2H", 1, 0),
    ("2H", 1, 1),  # away goal
    ("2H", 1, 1),
    ("2H", 2, 1),  # home goal
    ("FT", 2, 1),
]


def seed_fixture_and_history(
    conn: psycopg.Connection, fixture_id: int, home_name: str, away_name: str
) -> None:
    """Writes fake completed fixtures for both teams (so build_match_features
    has enough history to compute a pre-match prior) and the live fixture
    itself, starting just now so it reads as "in progress"."""
    starting_at = datetime.now(timezone.utc)
    history_fixture_id = fixture_id * 100

    with conn.cursor() as cur:
        for team_id, team_name in [
            (FAKE_HOME_TEAM_ID, home_name),
            (FAKE_AWAY_TEAM_ID, away_name),
        ]:
            for i in range(HISTORY_FIXTURES_PER_TEAM):
                history_fixture_id += 1
                is_home = i % 2 == 0
                goals_for, goals_against = (2, 1) if i % 3 else (1, 1)
                cur.execute(
                    """
                    INSERT INTO fixtures (
                        fixture_id, league_id, starting_at, home_id, away_id,
                        home_name, away_name, home_goals, away_goals, result, raw
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (fixture_id) DO NOTHING
                    """,
                    (
                        history_fixture_id,
                        1,
                        starting_at - timedelta(days=HISTORY_FIXTURES_PER_TEAM - i),
                        team_id if is_home else FAKE_FILLER_OPPONENT_ID,
                        FAKE_FILLER_OPPONENT_ID if is_home else team_id,
                        team_name if is_home else "Filler FC",
                        "Filler FC" if is_home else team_name,
                        goals_for if is_home else goals_against,
                        goals_against if is_home else goals_for,
                        "H" if is_home else "A",
                        Json({"statistics": []}),
                    ),
                )

        cur.execute(
            """
            INSERT INTO fixtures (
                fixture_id, league_id, starting_at, home_id, away_id,
                home_name, away_name, home_goals, away_goals, result, raw
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, 0, 0, 'NS', %s)
            ON CONFLICT (fixture_id) DO UPDATE SET starting_at = EXCLUDED.starting_at
            """,
            (
                fixture_id,
                1,
                starting_at,
                FAKE_HOME_TEAM_ID,
                FAKE_AWAY_TEAM_ID,
                home_name,
                away_name,
                Json({"statistics": []}),
            ),
        )
    conn.commit()


def tick_live_state(
    conn: psycopg.Connection, fixture_id: int, state: str, home_goals: int, away_goals: int
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO live_match_state (fixture_id, state, home_goals, away_goals)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (fixture_id) DO UPDATE SET
                state = EXCLUDED.state,
                home_goals = EXCLUDED.home_goals,
                away_goals = EXCLUDED.away_goals,
                updated_at = now()
            """,
            (fixture_id, state, home_goals, away_goals),
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture-id", type=int, default=900000)
    parser.add_argument("--home-name", default="Fake Home FC")
    parser.add_argument("--away-name", default="Fake Away FC")
    parser.add_argument(
        "--tick-seconds", type=float, default=4.0, help="seconds between state ticks"
    )
    args = parser.parse_args()

    with psycopg.connect(POSTGRES_URL) as conn:
        seed_fixture_and_history(conn, args.fixture_id, args.home_name, args.away_name)
        print(f"seeded fixture {args.fixture_id}: {args.home_name} vs {args.away_name}")
        print(f"dashboard: enter fixture id {args.fixture_id} and click Watch")
        print(f"raw ws: ws://localhost:8000/ws/predictions/{args.fixture_id}/live\n")

        for state, home_goals, away_goals in MATCH_SCRIPT:
            tick_live_state(conn, args.fixture_id, state, home_goals, away_goals)
            print(f"  -> {state} {home_goals}-{away_goals}")
            time.sleep(args.tick_seconds)

    print("\nmatch finished (FT) - re-run to replay, or pass a different --fixture-id")


if __name__ == "__main__":
    main()
