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


def get_team_history(team_id: int, before: datetime, limit: int) -> list[TeamResultRow]:
    """Most-recent-last list of a team's prior *completed* results strictly
    before `before`, for feeding into feature_lib.team_form().

    Deliberately *not* season-scoped, which it was until ml/FINDINGS_SCOPE.md
    measured what the scoping cost. It was the train/serve skew in this
    project: ml/build_features.py has always taken a club's last MAX_WINDOW
    matches whatever season they fall in, so a season-scoped query here fed
    the model an input distribution it was never fitted on - and refused 91%
    of August outright, because no club has MIN_WINDOW matches in a season
    that has just started.

    The ablation (ml/scope_ablate.py, rolling origin, 19,810 fixtures) found
    the two windows to be the same feature wherever both exist: -0.00001 log
    loss [-0.00058, +0.00057], and +0.00078 [-0.00329, +0.00512] on the
    August/September fixtures where the scoped window is a partial one. So
    the scoping bought no accuracy and cost 8.3% of all fixtures, which is
    the entire -0.00707 between the two arms once a refusal is priced at the
    class base rate.

    Rows without a final score (e.g. still-unbackfilled upcoming-fixture
    placeholders) are excluded so MIN_WINDOW only counts usable history.
    """
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT home_id, home_goals, away_goals, raw
            FROM fixtures
            WHERE (home_id = %s OR away_id = %s) AND starting_at < %s
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
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


def get_team_goal_volume(team_id: int, before: datetime, window: int) -> float:
    """A team's total goals scored over its last `window` completed fixtures
    strictly before `before` - feeds feature_lib.club_goals_relative(). Not
    season-scoped, and now neither is get_team_history(): the v2 model's
    GOALS_WINDOW (38) is close to a full season on purpose, so it's meant to
    span a season boundary early in a new one rather than reset to nothing."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT CASE WHEN home_id = %s THEN home_goals ELSE away_goals END AS goals_for
            FROM fixtures
            WHERE (home_id = %s OR away_id = %s) AND starting_at < %s
              AND home_goals IS NOT NULL AND away_goals IS NOT NULL
            ORDER BY starting_at DESC
            LIMIT %s
            """,
            (team_id, team_id, team_id, before, window),
        )
        return float(sum(row[0] for row in cur.fetchall()))


def get_league_goal_volume_reference(league_id: int, before: datetime, window: int) -> float | None:
    """The league's current cross-sectional mean of `get_team_goal_volume`
    across the teams that have played in it before `before` - the serving-time
    stand-in for feature_lib.LeagueRollingMean's rolling-200-historical-value
    reference used at training time.

    The two aren't the same computation, and that's fine here specifically:
    LeagueRollingMean exists to strip out an archive-density trend that only
    exists when scoring *past* dates across 25 years of backfill. Serving only
    ever asks about "now", where every team's own recent matches are already
    fully populated, so a live average across the league's current teams needs
    no such correction and is far cheaper than replaying the historical walk
    on every request."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            WITH teams_in_league AS (
                SELECT DISTINCT team_id FROM (
                    SELECT home_id AS team_id FROM fixtures
                    WHERE league_id = %(league_id)s AND starting_at < %(before)s
                    UNION
                    SELECT away_id AS team_id FROM fixtures
                    WHERE league_id = %(league_id)s AND starting_at < %(before)s
                ) t
            ),
            ranked AS (
                SELECT
                    tl.team_id,
                    CASE WHEN f.home_id = tl.team_id THEN f.home_goals ELSE f.away_goals END
                        AS goals_for,
                    ROW_NUMBER() OVER (
                        PARTITION BY tl.team_id ORDER BY f.starting_at DESC
                    ) AS rn
                FROM teams_in_league tl
                JOIN fixtures f
                    ON (f.home_id = tl.team_id OR f.away_id = tl.team_id)
                   AND f.starting_at < %(before)s
                   AND f.home_goals IS NOT NULL AND f.away_goals IS NOT NULL
            ),
            team_volumes AS (
                SELECT team_id, SUM(goals_for) AS volume
                FROM ranked
                WHERE rn <= %(window)s
                GROUP BY team_id
            )
            SELECT AVG(volume) FROM team_volumes
            """,
            {"league_id": league_id, "before": before, "window": window},
        )
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None


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
    model_version: str,
) -> None:
    """Appends a snapshot of model vs. market probabilities to `edges`, so
    backtest/grade_calls.py has real history to grade against outcomes
    (PROJECT_SPEC.md Phase 4). `prices` carries the decimal odds at this
    snapshot (home_win/draw/away_win keys) so a graded prediction_call can
    later simulate P&L against what was actually available, not the closing
    price. `model_version` tags which model produced model_probs (config's
    SERVED_MODEL_VERSION) so grading can never mix two models' records - the
    column has no DB default, so omitting it is an error, not a silent 'v1'."""
    with pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edges (
                fixture_id, model_home, model_draw, model_away,
                market_home, market_draw, market_away,
                edge_home, edge_draw, edge_away, flagged,
                home_price, draw_price, away_price, model_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                fixture_id,
                model_probs["home_win"], model_probs["draw"], model_probs["away_win"],
                market_probs["home_win"], market_probs["draw"], market_probs["away_win"],
                edge["home_win"], edge["draw"], edge["away_win"],
                flagged,
                prices["home_win"], prices["draw"], prices["away_win"],
                model_version,
            ),
        )
