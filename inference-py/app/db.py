"""Postgres + DynamoDB access for the inference service - reads the
`fixtures` table (Postgres) and live match state (DynamoDB) that
ingestor-go writes to (see ingestor-go/internal/store/postgres.go and
ingestor-go/internal/live/dynamo_store.go)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

import boto3
from psycopg_pool import ConnectionPool

from app.config import AWS_REGION, DYNAMODB_TABLE_NAME, POSTGRES_URL
from feature_lib import extract_team_stats

pool = ConnectionPool(POSTGRES_URL, min_size=1, max_size=5, open=False)

_dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
_live_state_table = _dynamodb.Table(DYNAMODB_TABLE_NAME)


class FixtureRow(TypedDict):
    fixture_id: int
    league_id: int
    starting_at: datetime
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    season_id: int


class TeamResultRow(TypedDict):
    goals_for: int
    goals_against: int
    stats: dict


class LiveStateRow(TypedDict):
    state: str
    home_goals: int
    away_goals: int
    updated_at: datetime


class OddsRow(TypedDict):
    implied_home: float
    implied_draw: float
    implied_away: float
    home_price: float
    draw_price: float
    away_price: float


def get_fixture(fixture_id: int) -> FixtureRow | None:
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id, league_id, starting_at, home_id, away_id, home_name, away_name,
                   (raw ->> 'season_id')::bigint AS season_id
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
            "home_id", "away_id", "home_name", "away_name", "season_id",
        )
        return dict(zip(keys, row))  # type: ignore[return-value]


def get_team_history(
    team_id: int, before: datetime, limit: int, season_id: int
) -> list[TeamResultRow]:
    """Most-recent-last list of a team's prior *completed* results strictly
    before `before` within the given season, for feeding into
    feature_lib.team_form(). Scoped to `season_id` so history never bleeds
    back into a previous season - a team's form should only reflect the
    season it's currently playing, so a fresh season starts with no
    predictions until MIN_WINDOW matches have been played in it, by design.
    Rows without a final score (e.g. still-unbackfilled upcoming-fixture
    placeholders) are excluded so MIN_WINDOW only counts usable history."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT home_id, home_goals, away_goals, raw
            FROM fixtures
            WHERE (home_id = %s OR away_id = %s) AND starting_at < %s
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
              AND (raw ->> 'season_id')::bigint = %s
            ORDER BY starting_at DESC
            LIMIT %s
            """,
            (team_id, team_id, before, season_id, limit),
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
    (see ingestor-go/internal/live/dynamo_store.go). Returns None if the
    fixture has never been polled as live, or its item has aged out via
    DynamoDB TTL (see infra/modules/dynamodb-live-state)."""
    resp = _live_state_table.get_item(Key={"fixture_id": fixture_id})
    item = resp.get("Item")
    if item is None:
        return None
    return {
        "state": item["state"],
        "home_goals": int(item["home_goals"]),
        "away_goals": int(item["away_goals"]),
        "updated_at": datetime.fromtimestamp(int(item["updated_at"]), tz=timezone.utc),
    }


def get_odds(fixture_id: int) -> OddsRow | None:
    """Reads the latest de-vigged odds ingestor-go's odds poller wrote (see
    ingestor-go/internal/store/postgres.go's odds table). Returns None if
    the fixture hasn't been matched to an odds event yet."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT implied_home, implied_draw, implied_away,
                   home_price, draw_price, away_price
            FROM odds
            WHERE fixture_id = %s
            """,
            (fixture_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        # Postgres NUMERIC comes back as decimal.Decimal, which can't be
        # mixed with the plain floats model.predict_proba() returns.
        keys = (
            "implied_home", "implied_draw", "implied_away",
            "home_price", "draw_price", "away_price",
        )
        return dict(zip(keys, (float(v) for v in row)))  # type: ignore[return-value]


def insert_edge(
    fixture_id: int,
    model_probs: dict,
    market_probs: dict,
    edge: dict,
    flagged: bool,
    prices: dict,
) -> None:
    """Appends a snapshot of model vs. market probabilities to `edges`, so
    backtest/grade_calls.py has real history to grade against outcomes
    (PROJECT_SPEC.md Phase 4). `prices` carries the decimal odds at this
    snapshot (home_win/draw/away_win keys) so a graded prediction_call can
    later simulate P&L against what was actually available, not the closing
    price."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edges (
                fixture_id, model_home, model_draw, model_away,
                market_home, market_draw, market_away,
                edge_home, edge_draw, edge_away, flagged,
                home_price, draw_price, away_price
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                fixture_id,
                model_probs["home_win"], model_probs["draw"], model_probs["away_win"],
                market_probs["home_win"], market_probs["draw"], market_probs["away_win"],
                edge["home_win"], edge["draw"], edge["away_win"],
                flagged,
                prices["home_win"], prices["draw"], prices["away_win"],
            ),
        )
