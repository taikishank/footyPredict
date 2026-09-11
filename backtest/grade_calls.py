"""Grades finished fixtures against the model's flagged edge calls.

For each fixture whose result is now known and that doesn't yet have a
prediction_calls row, picks the flagged `edges` snapshot closest to kickoff
as "the call" (edges is an append-only poll-by-poll log, so a fixture can
flip in and out of flagged as odds move before kickoff), compares it to what
actually happened, and records a graded row: whether the edge side won, the
Brier score of the full model (and market) distribution, and a flat-stake
paper P&L against the price recorded at that snapshot. Fixtures the model
never flagged an edge on are left ungraded entirely - see PROJECT_SPEC.md
Phase 6.

Run manually or on a schedule (cron); safe to re-run since already-graded
fixtures are excluded by the ungraded-fixtures query and the insert's
ON CONFLICT guard.
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

# Buffer after kickoff before treating a fixture as done - gives ingestor-go's
# batch pipeline time to backfill the final score before we go looking for it.
_GRADE_AFTER = timedelta(hours=3)

_OUTCOMES = ("home_win", "draw", "away_win")
_EDGE_COL = {"home_win": "edge_home", "draw": "edge_draw", "away_win": "edge_away"}
_MODEL_COL = {"home_win": "model_home", "draw": "model_draw", "away_win": "model_away"}
_MARKET_COL = {"home_win": "market_home", "draw": "market_draw", "away_win": "market_away"}
_PRICE_COL = {"home_win": "home_price", "draw": "draw_price", "away_win": "away_price"}


def find_ungraded_finished_fixtures(conn: psycopg.Connection) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.fixture_id
            FROM fixtures f
            WHERE f.result IS NOT NULL
              AND f.starting_at < now() - %s
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_calls pc WHERE pc.fixture_id = f.fixture_id
              )
            """,
            (_GRADE_AFTER,),
        )
        return [row[0] for row in cur.fetchall()]


def latest_flagged_edge(
    conn: psycopg.Connection, fixture_id: int, starting_at
) -> dict | None:
    """The flagged snapshot closest to kickoff - the last thing the model
    believed before the match was no longer "upcoming"."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT computed_at, model_home, model_draw, model_away,
                   market_home, market_draw, market_away,
                   edge_home, edge_draw, edge_away,
                   home_price, draw_price, away_price
            FROM edges
            WHERE fixture_id = %s AND flagged = true AND computed_at < %s
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (fixture_id, starting_at),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = (
            "computed_at", "model_home", "model_draw", "model_away",
            "market_home", "market_draw", "market_away",
            "edge_home", "edge_draw", "edge_away",
            "home_price", "draw_price", "away_price",
        )
        return dict(zip(keys, row))


def _brier_score(probs_by_outcome: dict[str, float], actual_result: str) -> float:
    return sum(
        (probs_by_outcome[o] - (1.0 if o == actual_result else 0.0)) ** 2
        for o in _OUTCOMES
    )


def grade_fixture(conn: psycopg.Connection, fixture_id: int) -> bool:
    """Grades one fixture's call, if it ever had a flagged edge. Returns
    True if a prediction_calls row was written."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT starting_at, result FROM fixtures WHERE fixture_id = %s",
            (fixture_id,),
        )
        starting_at, actual_result = cur.fetchone()

    call = latest_flagged_edge(conn, fixture_id, starting_at)
    if call is None:
        return False  # model never flagged an edge on this fixture

    call_side = max(_OUTCOMES, key=lambda o: abs(float(call[_EDGE_COL[o]])))
    model_prob = float(call[_MODEL_COL[call_side]])
    market_prob = float(call[_MARKET_COL[call_side]])
    edge = float(call[_EDGE_COL[call_side]])
    price = call[_PRICE_COL[call_side]]
    price = float(price) if price is not None else None

    correct = call_side == actual_result

    model_probs = {o: float(call[_MODEL_COL[o]]) for o in _OUTCOMES}
    market_probs = {o: float(call[_MARKET_COL[o]]) for o in _OUTCOMES}
    brier_score = _brier_score(model_probs, actual_result)
    market_brier_score = _brier_score(market_probs, actual_result)

    staked_units = 1.0
    pnl_units = None
    if price is not None:
        pnl_units = ((price - 1.0) if correct else -1.0) * staked_units

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_calls (
                fixture_id, edge_computed_at, call_side, model_prob, market_prob,
                edge, price, actual_result, correct, brier_score, market_brier_score,
                staked_units, pnl_units
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fixture_id) DO NOTHING
            """,
            (
                fixture_id, call["computed_at"], call_side, model_prob, market_prob,
                edge, price, actual_result, correct, brier_score, market_brier_score,
                staked_units, pnl_units,
            ),
        )
    return True


def main() -> None:
    graded = 0
    skipped = 0
    with psycopg.connect(POSTGRES_URL) as conn:
        for fixture_id in find_ungraded_finished_fixtures(conn):
            if grade_fixture(conn, fixture_id):
                graded += 1
            else:
                skipped += 1
        conn.commit()
    print(
        f"graded {graded} call(s); {skipped} finished fixture(s) had no flagged edge",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
