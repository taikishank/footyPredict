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

Prints the same summary a second time for double_chance_calls - the same
graded calls reframed as a double-chance bet (see grade_calls.py) - so the
straight single-outcome pick and its lower-payout, easier-to-win sibling can
be compared side by side.
"""
from __future__ import annotations

import os

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

_EDGE_BUCKETS = [(0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.01)]

# Tables sharing the exact same graded-call shape (fixture_id, model_prob,
# market_prob, edge, correct, brier_score, market_brier_score, pnl_units) -
# prediction_calls is the straight single-outcome pick, double_chance_calls
# its reframing as the pair of outcomes that excludes it. Names are fixed
# literals below, never user input, so f-string interpolation is safe here.
_CALL_TABLES = [
    ("single-outcome pick", "prediction_calls"),
    ("double-chance pick", "double_chance_calls"),
]


def _print_summary(cur, label: str, table: str) -> None:
    cur.execute(f"SELECT count(*) FROM {table}")
    (total,) = cur.fetchone()
    print(f"\n=== {label} ({table}) ===")
    if total == 0:
        print("no graded calls yet - run backtest/grade_calls.py first")
        return

    cur.execute(
        f"""
        SELECT
            avg(correct::int)::float,
            avg(model_prob)::float,
            avg(brier_score)::float,
            avg(market_brier_score)::float,
            sum(pnl_units) FILTER (WHERE pnl_units IS NOT NULL),
            count(*) FILTER (WHERE pnl_units IS NOT NULL)
        FROM {table}
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

    print("hit rate by edge size:")
    for lo, hi in _EDGE_BUCKETS:
        cur.execute(
            f"""
            SELECT count(*), avg(correct::int)::float
            FROM {table}
            WHERE abs(edge) >= %s AND abs(edge) < %s
            """,
            (lo, hi),
        )
        n, bucket_hit_rate = cur.fetchone()
        label_bucket = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">={lo:.0%}"
        if n:
            print(f"  {label_bucket:>8}: {bucket_hit_rate:.1%}  (n={n})")
        else:
            print(f"  {label_bucket:>8}: no calls yet")


def main() -> None:
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        for label, table in _CALL_TABLES:
            _print_summary(cur, label, table)


if __name__ == "__main__":
    main()
