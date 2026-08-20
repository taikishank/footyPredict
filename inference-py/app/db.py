"""Postgres access for the inference service - reads the `fixtures` table
that ingestor-go writes to (see ingestor-go/internal/store/postgres.go)."""
from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from psycopg_pool import ConnectionPool

from app.config import POSTGRES_URL
from feature_lib import extract_team_stats

pool = ConnectionPool(POSTGRES_URL, min_size=1, max_size=5, open=False)


class FixtureRow(TypedDict):
    fixture_id: int
    league_id: int
    starting_at: datetime
    home_id: int
    away_id: int
    home_name: str
    away_name: str


class TeamResultRow(TypedDict):
    goals_for: int
    goals_against: int
    stats: dict


class LiveStateRow(TypedDict):
    state: str
    home_goals: int
    away_goals: int
    updated_at: datetime


def get_fixture(fixture_id: int) -> FixtureRow | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id, league_id, starting_at, home_id, away_id, home_name, away_name
            FROM fixtures
            WHERE fixture_id = %s
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = (
            "fixture_id", "league_id", "starting_at",
            "home_id", "away_id", "home_name", "away_name",
        )
        return dict(zip(keys, row))  # type: ignore[return-value]


def get_team_history(team_id: int, before: datetime, limit: int) -> list[TeamResultRow]:
    """Most-recent-last list of a team's prior results strictly before
    `before`, for feeding into feature_lib.team_form()."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT home_id, home_goals, away_goals, raw
            FROM fixtures
            WHERE (home_id = %s OR away_id = %s) AND starting_at < %s
            ORDER BY starting_at DESC
            LIMIT %s
            """,
            (team_id, team_id, before, limit),
        )
        rows = cur.fetchall()

    history = []
    for home_id, home_goals, away_goals, raw in rows:
        is_home = home_id == team_id
        history.append({
            "goals_for": home_goals if is_home else away_goals,
            "goals_against": away_goals if is_home else home_goals,
            "stats": extract_team_stats(raw, team_id),
        })
    history.reverse()  # oldest -> most recent, matching team_form()'s expectation
    return history


def list_upcoming_fixtures(start: datetime, end: datetime) -> list[FixtureRow]:
    """Fixtures kicking off in [start, end), soonest first (see
    PROJECT_SPEC.md Phase 4's Upcoming tab)."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id, league_id, starting_at, home_id, away_id, home_name, away_name
            FROM fixtures
            WHERE starting_at >= %s AND starting_at < %s
            ORDER BY starting_at ASC
            """,
            (start, end),
        )
        rows = cur.fetchall()
        keys = (
            "fixture_id", "league_id", "starting_at",
            "home_id", "away_id", "home_name", "away_name",
        )
        return [dict(zip(keys, row)) for row in rows]  # type: ignore[misc]


def get_live_state(fixture_id: int) -> LiveStateRow | None:
    """Reads the latest polled live state ingestor-go's live poller wrote
    (see ingestor-go/internal/store/postgres.go's live_match_state table).
    Returns None if the fixture has never been polled as live."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT state, home_goals, away_goals, updated_at
            FROM live_match_state
            WHERE fixture_id = %s
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = ("state", "home_goals", "away_goals", "updated_at")
        return dict(zip(keys, row))  # type: ignore[return-value]
