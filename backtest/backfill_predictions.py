"""Reconstructs v2 edges snapshots for fixtures that have already kicked off,
so grading doesn't have to wait for live polling to accumulate a record.

Why this is legitimate rather than cheating, and the two things that would
make it cheating:

1. Features are point-in-time by construction. app/db.py's get_team_history
   bounds every window at `starting_at < <this fixture's kickoff>` and only
   counts completed results, so the vector built now is the same vector that
   would have been built an hour before kickoff. Nothing about the fixture's
   own result reaches it.
2. Odds are whatever the poller last wrote before kickoff (verified: no odds
   row in this window was fetched after its fixture started). That's a real
   price that was really available, close to closing.
3. The model must not have trained on the fixture. THIS is the one that
   silently invalidates everything, so it is enforced, not trusted: the
   script refuses to touch any fixture at or before the last date in v2's
   training file (ml/train_v2.py's FEATURES_PATH). Grading a model on rows
   it fit - or even picked hyperparameters against - measures memory, not
   skill.

Snapshots are written with computed_at = the odds row's fetched_at, not
now(): that is the moment the inputs were actually valid, and
backtest/grade_calls.py only considers snapshots with computed_at < kickoff.
A now() timestamp would make every backfilled row invisible to grading.

They are tagged model_version 'v2-backfill' (--model-version to override) so
they never blend into the live v2 record. Reconstructed and live snapshots
answer different questions - live proves the serving pipeline worked at the
time, backfill only proves what the model would have said - and comparing
the two once live data accumulates is a free check that the pipeline agrees
with the model.

    python backtest/backfill_predictions.py --hours 48 [--dry-run]
    python backtest/grade_calls.py --model v2-backfill
    python backtest/stats.py --model v2-backfill
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from dotenv import find_dotenv, load_dotenv

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "inference-py"))

load_dotenv(find_dotenv(usecwd=True))

from app import db, model_v2  # noqa: E402
from app.config import EDGE_FLAG_THRESHOLD  # noqa: E402
from app.features import NotEnoughHistoryError, build_match_features  # noqa: E402

POSTGRES_URL = os.environ["POSTGRES_URL"]

DEFAULT_MODEL_VERSION = "v2-backfill"
_TRAINING_FEATURES = _REPO / "ml" / "data" / "features" / "2000_window3_8.csv"


def training_data_end():
    """Last kickoff date v2 was fitted or validated against. Read straight
    from ml/train_v2.py's feature file rather than hardcoded, so a retrain on
    fresher data moves the guard automatically instead of silently leaving a
    stale cutoff behind."""
    import pandas as pd

    dates = pd.read_csv(_TRAINING_FEATURES, usecols=["date"], parse_dates=["date"])
    return dates["date"].max().to_pydatetime()


def _seen_in_training(kickoff, cutoff) -> bool:
    """Kickoffs come back tz-aware from Postgres; the training file's dates
    are naive local times. Compare on naive wall-clock - the cutoff is a
    coarse safety rail, not a millisecond boundary."""
    return kickoff.replace(tzinfo=None) <= cutoff


def fixtures_to_backfill(conn: psycopg.Connection, hours: int, model_version: str):
    """Kicked-off fixtures in the window that have odds and no snapshot for
    this model version yet. Ordered oldest-first so a partial run is a clean
    prefix of a full one."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.fixture_id, f.starting_at, o.fetched_at,
                   o.implied_home, o.implied_draw, o.implied_away,
                   o.home_price, o.draw_price, o.away_price
            FROM fixtures f
            JOIN odds o USING (fixture_id)
            WHERE f.starting_at >= now() - make_interval(hours => %s)
              AND f.starting_at < now()
              AND o.fetched_at < f.starting_at
              AND NOT EXISTS (
                  SELECT 1 FROM edges e
                  WHERE e.fixture_id = f.fixture_id AND e.model_version = %s
              )
            ORDER BY f.starting_at
            """,
            (hours, model_version),
        )
        keys = (
            "fixture_id", "starting_at", "fetched_at",
            "implied_home", "implied_draw", "implied_away",
            "home_price", "draw_price", "away_price",
        )
        return [dict(zip(keys, row)) for row in cur.fetchall()]


def insert_backfilled_edge(
    conn: psycopg.Connection, row: dict, model_probs: dict, model_version: str
) -> None:
    market = {
        "home_win": float(row["implied_home"]),
        "draw": float(row["implied_draw"]),
        "away_win": float(row["implied_away"]),
    }
    edge = {o: model_probs[o] - market[o] for o in market}
    flagged = max(abs(v) for v in edge.values()) >= EDGE_FLAG_THRESHOLD

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO edges (
                fixture_id, computed_at, model_home, model_draw, model_away,
                market_home, market_draw, market_away,
                edge_home, edge_draw, edge_away, flagged,
                home_price, draw_price, away_price, model_version
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fixture_id, computed_at) DO NOTHING
            """,
            (
                row["fixture_id"], row["fetched_at"],
                model_probs["home_win"], model_probs["draw"], model_probs["away_win"],
                market["home_win"], market["draw"], market["away_win"],
                edge["home_win"], edge["draw"], edge["away_win"], flagged,
                row["home_price"], row["draw_price"], row["away_price"],
                model_version,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct model snapshots for already-played fixtures.",
    )
    parser.add_argument(
        "--hours", type=int, default=48, help="look back this many hours (default: 48)"
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"edges.model_version tag to write (default: {DEFAULT_MODEL_VERSION})",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written, write nothing"
    )
    args = parser.parse_args()

    cutoff = training_data_end()
    print(f"v2 training data ends {cutoff} - fixtures at or before it are refused", file=sys.stderr)

    db.pool.open()
    model_v2.load_model()
    written = skipped_seen = skipped_history = 0
    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            rows = fixtures_to_backfill(conn, args.hours, args.model_version)
            print(f"{len(rows)} fixture(s) in window with odds and no {args.model_version} snapshot",
                  file=sys.stderr)

            for row in rows:
                if _seen_in_training(row["starting_at"], cutoff):
                    skipped_seen += 1
                    continue

                try:
                    features = build_match_features(row["fixture_id"])
                except NotEnoughHistoryError:
                    skipped_history += 1
                    continue
                if features is None or features.v2_vector is None:
                    skipped_history += 1
                    continue

                probs = model_v2.predict_proba(features.v2_vector)
                if args.dry_run:
                    written += 1
                    continue
                insert_backfilled_edge(conn, row, probs, args.model_version)
                written += 1

            if not args.dry_run:
                conn.commit()
    finally:
        db.pool.close()

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"[{args.model_version}] {verb} {written} snapshot(s); "
        f"skipped {skipped_seen} inside training data, "
        f"{skipped_history} without enough history",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
