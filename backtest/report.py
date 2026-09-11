"""Summarizes real-life model performance on flagged edges, from the graded
rows backtest/grade_calls.py writes to prediction_calls. Run after
grade_calls.py; read-only.

Prints:
- hit rate on the edge side, and the model's own average confidence in that
  side (compare the two - a well-calibrated model's hit rate should track
  its average predicted probability, not just beat 50%)
- mean Brier score for flagged calls vs. the market's Brier score on the
  same fixtures - the real test of whether a flagged edge is signal rather
  than noise: is the model *more* accurate than the market where it claims
  to disagree with it?
- flat-stake paper P&L (1 unit per call), for fixtures with a recorded price
- hit rate broken out by edge-size bucket, since a model claiming a 20-point
  edge should be right far more often than one claiming a 5-point edge if
  its probabilities are meaningful
"""
from __future__ import annotations

import os

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

_EDGE_BUCKETS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.01)]


def main() -> None:
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM prediction_calls")
        (total,) = cur.fetchone()
        if total == 0:
            print("no graded calls yet - run backtest/grade_calls.py first")
            return

        cur.execute(
            """
            SELECT
                avg(correct::int)::float,
                avg(model_prob)::float,
                avg(brier_score)::float,
                avg(market_brier_score)::float,
                sum(pnl_units) FILTER (WHERE pnl_units IS NOT NULL),
                count(*) FILTER (WHERE pnl_units IS NOT NULL)
            FROM prediction_calls
            """
        )
        hit_rate, avg_conf, brier, market_brier, pnl, priced = cur.fetchone()

        print(f"flagged calls graded: {total}")
        print(f"hit rate:             {hit_rate:.1%}  (avg model confidence: {avg_conf:.1%})")
        print(f"model brier score:    {brier:.4f}  (lower is better)")
        print(f"market brier score:   {market_brier:.4f}  (edge over market: {market_brier - brier:+.4f})")
        if priced:
            print(f"paper P&L:            {pnl:+.2f} units over {priced} priced call(s)")
        else:
            print("paper P&L:            no priced calls yet")

        print("\nhit rate by edge size:")
        for lo, hi in _EDGE_BUCKETS:
            cur.execute(
                """
                SELECT count(*), avg(correct::int)::float
                FROM prediction_calls
                WHERE abs(edge) >= %s AND abs(edge) < %s
                """,
                (lo, hi),
            )
            n, bucket_hit_rate = cur.fetchone()
            label = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">={lo:.0%}"
            if n:
                print(f"  {label:>8}: {bucket_hit_rate:.1%}  (n={n})")
            else:
                print(f"  {label:>8}: no calls yet")


if __name__ == "__main__":
    main()
