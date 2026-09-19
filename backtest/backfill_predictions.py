"""Reconstructs edges snapshots for fixtures that have already kicked off,
so grading doesn't have to wait for live polling to accumulate a record.

Why this is legitimate rather than cheating, and the three things that would
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
   script refuses to touch any fixture at or before the last date in the
   served model's training table. Grading a model on rows it fit - or even
   picked hyperparameters against - measures memory, not skill.
4. **v3 only, and the reason this script could not simply be pointed at v3:**
   v3's seven rating columns are not point-in-time. Condition 1 holds for
   every column that is a bounded query, and a rating is not a query -
   app/ratings_state.py replays the *whole* archive, so the state it serves
   for a fixture on 12 September already contains that fixture's own result
   and every result after it. Building a v3 vector through the live singleton
   would hand the model the answer and produce a backtest that looks
   extraordinary and means nothing.

   So the rating columns are replayed per fixture instead: point_in_time_
   ratings() walks the archive in (starting_at, fixture_id) order - the same
   driver and the same order as ml/build_rating_features.build() - and reads
   each target fixture's features() strictly *before* record()s its result.
   That is the identical row the offline build emitted for it, which is the
   definition of pre-match. The live ratings_state singleton is deliberately
   never loaded here.

Snapshots are written with computed_at = the odds row's fetched_at, not
now(): that is the moment the inputs were actually valid, and
backtest/grade_calls.py only considers snapshots with computed_at < kickoff.
A now() timestamp would make every backfilled row invisible to grading.

They are tagged '<model>-backfill' (--model-version to override) so they never
blend into the live record. Reconstructed and live snapshots answer different
questions - live proves the serving pipeline worked at the time, backfill only
proves what the model would have said - and comparing the two once live data
accumulates is a free check that the pipeline agrees with the model.

--model defaults to whatever inference-py serves, for the same reason
grade_calls.py and report.py do: after a promotion the record worth
reconstructing is the promoted model's, not the retired one's.

    python backtest/backfill_predictions.py --hours 48 [--dry-run]
    python backtest/backfill_predictions.py --model v2 --hours 48
    python backtest/grade_calls.py --model v3-backfill
    python backtest/stats.py --model v3-backfill

Reconstructing both tiers over the same fixtures is what makes them directly
comparable, which is what backtest/stats.py --compare wants:

    python backtest/backfill_predictions.py --model v2 --hours 72
    python backtest/backfill_predictions.py --model v3 --hours 72
    python backtest/grade_calls.py --model v2-backfill
    python backtest/grade_calls.py --model v3-backfill
    python backtest/stats.py --compare v2-backfill v3-backfill
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import psycopg
from dotenv import find_dotenv, load_dotenv

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "inference-py"))

load_dotenv(find_dotenv(usecwd=True))

from app import db, model_v2, model_v3, ratings_state  # noqa: E402
from app.config import EDGE_FLAG_THRESHOLD, SERVED_MODEL_VERSION  # noqa: E402
from app.features import NotEnoughHistoryError, build_match_features  # noqa: E402

POSTGRES_URL = os.environ["POSTGRES_URL"]

#: Reconstruct the served model's record by default - see the module docstring.
DEFAULT_MODEL = SERVED_MODEL_VERSION

#: model -> (loader module, the MatchFeatures attribute its vector starts from,
#: whether it needs point-in-time rating columns appended).
_MODELS = {
    "v2": (model_v2, "v2_vector", False),
    "v3": (model_v3, "v2_vector", True),
}

_TRAINING_FEATURES = _REPO / "ml" / "data" / "features" / "2000_window3_8.csv"
_TRAINING_RATINGS = _REPO / "ml" / "data" / "features" / "ratings.csv"


def training_data_end(model: str):
    """Last kickoff date the model was fitted or validated against.

    Read from the training files rather than hardcoded, so a retrain on fresher
    data moves the guard automatically instead of silently leaving a stale
    cutoff behind.

    For v3 the table is ml/rating_ablate.load()'s *inner join* of the form
    features and the rating rows, so its last date is the earlier of the two
    files' - taking the ratings file alone would put the cutoff a day late and
    wave through fixtures v3 had actually seen. Computed as the min of the two
    maxima rather than by doing the join, which would read 50MB to learn one
    date.
    """
    import pandas as pd

    def last(path):
        dates = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
        return dates["date"].max().to_pydatetime()

    if model == "v2":
        return last(_TRAINING_FEATURES)
    return min(last(_TRAINING_FEATURES), last(_TRAINING_RATINGS))


def point_in_time_ratings(conn: psycopg.Connection, targets: set[int]) -> dict[int, dict]:
    """Each target fixture's pre-match rating columns, replayed to it and no
    further.

    This is the leak check condition 4 exists for. The walk is
    ml/build_rating_features.build()'s, including its ordering and its
    features()-strictly-before-record() discipline, so what comes back is the
    row that build would have written for the fixture - by construction it
    cannot contain the fixture's own result, or any later one.

    One pass for all targets rather than one replay per fixture: the state is
    cumulative, so walking it once and reading off each target as it goes is
    the same answer for ~0.3s instead of 0.3s x however many fixtures.
    """
    from ratings_lib import ARMS, Replay

    with conn.cursor() as cur:
        # fixture_id is not in db.get_results_index()'s ResultRow, and the
        # ordering must match build_rating_features.order() exactly - see
        # app/ratings_state.py on why the tie-break is load-bearing.
        cur.execute(
            """
            SELECT fixture_id, league_id, starting_at, home_id, away_id,
                   home_goals, away_goals
            FROM fixtures
            WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
            ORDER BY starting_at ASC, fixture_id ASC
            """
        )
        rows = cur.fetchall()

    replay = Replay({ratings_state.ARM: ARMS[ratings_state.ARM]()})
    out: dict[int, dict] = {}
    for fixture_id, league_id, starting_at, home_id, away_id, home_goals, away_goals in rows:
        replay.roll_to(league_id, home_id, away_id, starting_at)
        if fixture_id in targets:
            out[fixture_id] = replay.features(home_id, away_id, league_id)
        replay.record(league_id, home_id, away_id, starting_at, home_goals, away_goals)
    return out


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
) -> bool:
    """Writes one snapshot. Returns False when the row was dropped.

    Returning the outcome rather than assuming it, because there is a live way
    for the insert to do nothing: edges' primary key is (fixture_id,
    computed_at) and does NOT include model_version, so reconstructing a second
    model over a fixture some other model already has a snapshot for at the
    same odds timestamp silently loses it. Counting attempts instead of rows
    reported 43 writes for 12 actual ones.
    """
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
            ON CONFLICT (fixture_id, computed_at, model_version) DO NOTHING
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
        return cur.rowcount > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct model snapshots for already-played fixtures.",
    )
    parser.add_argument(
        "--hours", type=int, default=48, help="look back this many hours (default: 48)"
    )
    parser.add_argument(
        "--model",
        choices=sorted(_MODELS),
        default=DEFAULT_MODEL,
        help=f"which model to reconstruct (default: {DEFAULT_MODEL}, the served one)",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="edges.model_version tag to write (default: <model>-backfill)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be written, write nothing"
    )
    args = parser.parse_args()

    module, vector_attr, needs_ratings = _MODELS[args.model]
    tag = args.model_version or f"{args.model}-backfill"

    cutoff = training_data_end(args.model)
    print(f"{args.model} training data ends {cutoff} - fixtures at or before it are refused",
          file=sys.stderr)

    db.pool.open()
    module.load_model()
    written = skipped_seen = skipped_history = skipped_rating = collided = 0
    try:
        with psycopg.connect(POSTGRES_URL) as conn:
            rows = fixtures_to_backfill(conn, args.hours, tag)
            print(f"{len(rows)} fixture(s) in window with odds and no {tag} snapshot",
                  file=sys.stderr)

            eligible = [r for r in rows if not _seen_in_training(r["starting_at"], cutoff)]
            skipped_seen = len(rows) - len(eligible)

            ratings = {}
            if needs_ratings and eligible:
                # Replayed here, once, for exactly the fixtures being written -
                # never read from the live ratings_state singleton, which has
                # already seen these results. See condition 4 in the docstring.
                ratings = point_in_time_ratings(
                    conn, {r["fixture_id"] for r in eligible})
                print(f"replayed pre-match {ratings_state.ARM} ratings for "
                      f"{len(ratings)}/{len(eligible)} fixture(s)", file=sys.stderr)

            for row in eligible:
                try:
                    features = build_match_features(row["fixture_id"])
                except NotEnoughHistoryError:
                    skipped_history += 1
                    continue
                if features is None or getattr(features, vector_attr) is None:
                    skipped_history += 1
                    continue

                vector = getattr(features, vector_attr)
                if needs_ratings:
                    rating_row = ratings.get(row["fixture_id"])
                    if rating_row is None:
                        # Not in the results index - an uncompleted fixture the
                        # replay never reached. Refuse rather than substitute.
                        skipped_rating += 1
                        continue
                    vector = np.hstack([
                        vector,
                        np.array([[rating_row[name]
                                   for name in ratings_state.FIELD_ORDER]], dtype=float),
                    ])

                probs = module.predict_proba(vector)
                if args.dry_run:
                    written += 1
                    continue
                if insert_backfilled_edge(conn, row, probs, tag):
                    written += 1
                else:
                    collided += 1

            if not args.dry_run:
                conn.commit()
    finally:
        db.pool.close()

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"[{tag}] {verb} {written} snapshot(s); "
        f"skipped {skipped_seen} inside training data, "
        f"{skipped_history} without enough history"
        + (f", {skipped_rating} without a replayed rating row" if skipped_rating else "")
        + (f"; {collided} DROPPED - another model already holds that "
           f"(fixture_id, computed_at); edges' primary key does not include "
           f"model_version" if collided else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
