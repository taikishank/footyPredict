"""v2 against v3 on the inputs the service will actually hand them.

Every published v3-over-v2 number in this project was measured on
ml/data/features/2000_window3_8.csv joined to ratings.csv - the training
table. That is the right table for choosing columns, and it is not the table
the service reads. scripts/verify_served_vector.py established how the two
differ, and the answer was not "not at all":

  * the 54 form columns are identical to 0.000000000, so nothing to do here;
  * the 7 rating columns are replayed from Postgres, and
    scripts/verify_archive_read.py proved that replay identical to the trained
    one over the whole shared prefix;
  * the 2 `club_goals_relative` columns differ by ~0.107 SD, because the
    league reference is a rolling historical mean at training time and a live
    cross-sectional one at serving time. That difference is deliberate and
    documented (db.get_league_goal_volume_reference), and it is the only place
    the served row departs from the trained row.

So the question this script answers is narrow and answerable: **does v3's
margin over v2 survive being fed the served goals columns instead of the
trained ones?**

It is not obvious either way. The skew is common-mode - both models read the
same two columns, with the same values - which argues it should largely cancel
in a paired comparison. But the models weight those columns differently (v3 has
seven rating columns competing for the same signal at a third of v2's depth),
so cancellation is a hypothesis, not an identity. Cheap to check, and the
alternative way of finding out is to promote and read it off graded rows weeks
later, with the fit change and the input change confounded.

Design:

  * Rows are the last VAL_FRACTION of the archive in date order - the
    chronological holdout neither shipped artifact was fitted on
    (ml/train_v2.py and ml/train_v3.py both fit the first 80%). Scoring an
    artifact on its own training rows would flatter the deeper model.
  * The shipped artifacts are loaded as-is, not refitted. This compares what
    would serve, not what a retrain would produce.
  * The two arms differ *only* in the two goals columns. Form and rating
    columns are held at their trained values, which the two verify scripts
    established are what serving produces anyway - so the comparison isolates
    the one column pair that actually moved.

Usage (Postgres running, with the archive backfilled):

    ./inference-py/bin/python inference-py/scripts/compare_v2_v3_served.py
    ./inference-py/bin/python inference-py/scripts/compare_v2_v3_served.py --limit 2000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from app import db
from app.config import MODEL_V2_LOCAL_PATH, MODEL_V3_LOCAL_PATH

import rating_ablate as ra
import train_v2 as tv2
import train_v3 as tv3
from feature_lib import GOALS_WINDOW, club_goals_relative

GOALS_COLUMNS = ["home_club_goals_relative", "away_club_goals_relative"]


def load_models() -> tuple[XGBClassifier, XGBClassifier]:
    """The shipped artifacts, exactly as app/model_v2.py and app/model_v3.py
    load them."""
    models = []
    for path in (MODEL_V2_LOCAL_PATH, MODEL_V3_LOCAL_PATH):
        if not path.exists():
            raise SystemExit(f"no artifact at {path} - run the matching ml/train_*.py")
        model = XGBClassifier()
        model.load_model(str(path))
        models.append(model)
    return models[0], models[1]


def served_goals_columns(df) -> np.ndarray:
    """Recompute the two goals columns the way the service would, per fixture.

    One league-reference query per fixture rather than one per league-day: the
    reference moves with `before`, and batching it by day would be a different
    quantity from the one build_match_features() asks for. At a few ms each
    this is the slow part of the script and still under a minute.
    """
    out = np.empty((len(df), 2), dtype=float)
    started = time.monotonic()
    for i, row in enumerate(df.itertuples(index=False)):
        fixture = db.get_fixture(int(row.fixture_id))
        if fixture is None:
            out[i] = np.nan
            continue
        at = fixture["starting_at"]
        reference = db.get_league_goal_volume_reference(
            fixture["league_id"], at, GOALS_WINDOW)
        for j, team_id in enumerate((fixture["home_id"], fixture["away_id"])):
            volume = db.get_team_goal_volume(team_id, at, GOALS_WINDOW)
            value = club_goals_relative(volume, reference)
            out[i, j] = np.nan if value is None else value
        if (i + 1) % 1000 == 0:
            print(f"  rebuilt {i + 1:,}/{len(df):,} "
                  f"({time.monotonic() - started:.0f}s)", flush=True)
    return out


def losses(model: XGBClassifier, X, y: np.ndarray) -> np.ndarray:
    """Per-fixture log loss, so the bootstrap can pair on the fixture."""
    probs = model.predict_proba(X)
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, None))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="score only the most recent N holdout fixtures (0 = all)")
    args = parser.parse_args()

    df, base_cols = ra.load()
    v3_cols = base_cols + ra.ARM_COLUMNS[tv3.RATING_ARM]
    y_all = df["result"].map(tv2.LABEL_MAP).to_numpy()

    cut = int(len(df) * (1 - tv2.VAL_FRACTION))
    holdout = df.iloc[cut:].reset_index(drop=True)
    y = y_all[cut:]
    if args.limit:
        holdout = holdout.iloc[-args.limit:].reset_index(drop=True)
        y = y[-args.limit:]

    print(f"holdout: {len(holdout):,} fixtures "
          f"({holdout['date'].min():%Y-%m-%d}..{holdout['date'].max():%Y-%m-%d})")
    print("neither shipped artifact was fitted on these rows\n")

    v2_model, v3_model = load_models()

    print("rebuilding the 2 goals columns through the serving path...")
    db.pool.open()
    try:
        served = served_goals_columns(holdout)
    finally:
        db.pool.close()

    trained = holdout[GOALS_COLUMNS].to_numpy(dtype=float)
    gap = np.abs(served - trained)
    finite = np.isfinite(gap)
    print(f"  mean |served - trained| on those columns: "
          f"{gap[finite].mean():.4f} "
          f"({gap[finite].mean() / holdout[GOALS_COLUMNS].stack().std():.3f} SD)")
    print(f"  fixtures where the served reference is unavailable: "
          f"{int((~np.isfinite(served)).any(axis=1).sum()):,}\n")

    # Two feature tables differing only in those two columns.
    results = {}
    for arm in ("trained", "served"):
        frame = holdout.copy()
        if arm == "served":
            frame[GOALS_COLUMNS] = served
        results[arm] = {
            "v2": losses(v2_model, frame[base_cols], y),
            "v3": losses(v3_model, frame[v3_cols], y),
        }

    # Header says "v3 gain", not "v3 - v2". paired_bootstrap returns
    # baseline - arm, so the number is v2's loss minus v3's: positive means v3
    # loses less, i.e. v3 is better. Labelling it as a difference invites
    # reading it as "v3 minus v2", which has the opposite sign convention to
    # the one it actually carries and makes the winner look like the loser.
    print(f"{'inputs':>10} {'v2 log loss':>12} {'v3 log loss':>12} "
          f"{'v3 gain':>10} {'95% CI':>22} {'P(v3)':>7}")
    print("-" * 78)
    for arm in ("trained", "served"):
        v2_losses, v3_losses = results[arm]["v2"], results[arm]["v3"]
        stats = ra.paired_bootstrap(v2_losses, v3_losses)
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{arm:>10} {v2_losses.mean():12.5f} {v3_losses.mean():12.5f} "
              f"{stats['improvement']:+10.5f} {ci:>22} {stats['p_better']:7.3f}")

    print("\nLower log loss is better, so a positive 'v3 gain' means v3 beats v2.")
    print("An interval spanning zero is a null result regardless of where the")
    print("point estimate sits.")

    # The question the script exists for: not whether v3 wins on either row,
    # but whether moving from trained inputs to served ones changes the margin
    # by more than the margin itself.
    trained_margin = ra.paired_bootstrap(results["trained"]["v2"],
                                         results["trained"]["v3"])["improvement"]
    served_margin = ra.paired_bootstrap(results["served"]["v2"],
                                        results["served"]["v3"])["improvement"]
    shift = served_margin - trained_margin
    print(f"\nv3's gain on trained inputs: {trained_margin:+.5f}")
    print(f"v3's gain on served inputs:  {served_margin:+.5f}")
    print(f"shift attributable to the goals-column skew: {shift:+.5f} "
          f"({abs(shift) / abs(trained_margin) * 100:.0f}% of the margin)")

    # Each model against itself across the two input sets - what the skew costs
    # in absolute terms, which the paired margin above deliberately cancels.
    print("\nwhat the skew costs each model on its own:")
    for name in ("v2", "v3"):
        delta = results["served"][name].mean() - results["trained"][name].mean()
        print(f"  {name}: {delta:+.5f} log loss on served inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
