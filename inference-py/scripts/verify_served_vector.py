"""Does the feature row the service builds equal the row the model trained on?

scripts/verify_archive_read.py closed the rating half of the train/serve
question: the archive Postgres holds is now the archive ml/ trained on, so
app/ratings_state.py replays the same history build_rating_features.py did.
This closes the other half, which the backfill made answerable for the first
time.

The 56 columns v2 and v3 share are not replayed state - they are queries.
build_match_features() computes them from `fixtures` per request; ml/
computed them once from the raw archive into
data/features/2000_window3_8.csv. Two implementations of the same definition,
never compared on a real fixture, because until the backfill Postgres held one
season and every fixture in the training table predated it. Now every training
row has a queryable counterpart.

This script picks fixtures from the training table, rebuilds their vectors
through the serving path, and diffs.

Two columns are *expected* to differ, and the point of measuring is to price
the difference rather than discover it:

  `{home,away}_club_goals_relative` divide a club's goal volume by a league
  reference, and the two sides compute that reference differently on purpose.
  Training uses feature_lib.LeagueRollingMean, a rolling mean over the last 200
  historical values, which exists to strip an archive-density trend that only
  appears when scoring past dates across 25 years of backfill. Serving uses
  db.get_league_goal_volume_reference(), a live cross-sectional mean over the
  league's current clubs - see its docstring for why that is the right call for
  "now" and why replaying the historical walk per request is not. On a *past*
  fixture the two therefore disagree by construction. How much they disagree
  has never been measured, and it bounds what this difference can cost the
  served prediction.

The seven rating columns are deliberately NOT compared here. They cannot be:
ratings_state has replayed the whole archive, so the state it would serve for a
2015 fixture already contains every result after it. That is correct in
production, where only upcoming fixtures are asked about, and it makes a
past-fixture comparison meaningless. Their equivalence is established
differently, by verify_archive_read.py's checks 3 and 4, which compare the two
replays as replays rather than per fixture.

Usage:

    ./inference-py/bin/python inference-py/scripts/verify_served_vector.py
    ./inference-py/bin/python inference-py/scripts/verify_served_vector.py --sample 200

Exit status is 0 when the 54 form columns agree within tolerance, 1 otherwise.
The two goals columns are reported but do not gate, for the reason above.
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app import db, features
from feature_lib import FORM_FIELD_ORDER

import rating_ablate as ra

#: Serving's v2 column order, from build_match_features(): home form fields,
#: away form fields, then the two goals columns. Asserted against the training
#: table's own order rather than assumed - a silent transposition here would
#: feed the model a permuted row that still has the right shape.
FORM_COLUMNS = ([f"home_form_{name}" for name in FORM_FIELD_ORDER]
                + [f"away_form_{name}" for name in FORM_FIELD_ORDER])
GOALS_COLUMNS = ["home_club_goals_relative", "away_club_goals_relative"]

#: Float noise between a CSV round-trip and a fresh computation.
TOLERANCE = 1e-6


def load_training_table() -> tuple[pd.DataFrame, list[str]]:
    """The exact table train_v3.py fits on, via rating_ablate.load()."""
    df, base_cols = ra.load()
    expected = FORM_COLUMNS + GOALS_COLUMNS
    if base_cols != expected:
        raise SystemExit(
            "training column order does not match the serving vector's:\n"
            f"  training: {base_cols[:4]} ... {base_cols[-2:]}\n"
            f"  serving:  {expected[:4]} ... {expected[-2:]}"
        )
    return df, base_cols


def pick(df: pd.DataFrame, sample: int, seed: int) -> pd.DataFrame:
    """A spread of fixtures rather than a block of them.

    Stratified by league and sampled across the whole date range, because the
    two failure modes this could find are league-shaped (a club whose history
    Postgres carries differently) and era-shaped (a season the backfill got
    wrong). A contiguous slice would see neither.
    """
    per_league = max(1, sample // df["league_id"].nunique())
    return (df.groupby("league_id", group_keys=False)
              .apply(lambda g: g.sample(min(per_league, len(g)), random_state=seed))
              .sort_values("date"))


def report_goals(diffs: np.ndarray, dates: pd.Series, df: pd.DataFrame) -> None:
    """What the goals columns' train/serve gap costs, in units of the column.

    Reported as a distribution in SD units rather than a max, and split by era,
    because the max is the wrong summary here and says so loudly: it is always
    drawn from the first years of the archive, where LeagueRollingMean is still
    filling its 200-value window and no definition of "currently in the league"
    can agree with a reference that is barely warm. Those fixtures are not the
    regime the service runs in. The last-365-days row is.
    """
    sd = float(df[GOALS_COLUMNS].std().mean())
    recent = dates >= dates.max() - timedelta(days=365)
    print(f"  column SD across the training table: {sd:.4f}")
    print(f"\n  {'era':>18} {'n':>5} {'mean':>8} {'SD units':>9} {'p90':>8} {'max':>8}")
    for label, mask in (("all eras", np.ones(len(diffs), dtype=bool)),
                        ("last 365 days", recent.to_numpy())):
        subset = diffs[mask]
        if not len(subset):
            continue
        print(f"  {label:>18} {len(subset):5d} {subset.mean():8.4f} "
              f"{subset.mean() / sd:9.3f} {np.percentile(subset, 90):8.4f} "
              f"{subset.max():8.4f}")
    print("\n  The league reference is a rolling mean over the last 200 historical")
    print("  values at training time and a live cross-sectional mean over the")
    print("  clubs currently in the league at serving time - two definitions, by")
    print("  design (db.get_league_goal_volume_reference). This does not gate;")
    print("  it prices the choice. For scale: before db.LEAGUE_REFERENCE_RECENCY")
    print("  bounded 'currently', the last-365-days figure was 0.630 SD.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=100,
                        help="fixtures to rebuild (spread across leagues)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df, base_cols = load_training_table()
    chosen = pick(df, args.sample, args.seed)
    print(f"comparing {len(chosen)} fixtures "
          f"({chosen['date'].min():%Y-%m-%d}..{chosen['date'].max():%Y-%m-%d})\n")

    db.pool.open()
    diffs: list[np.ndarray] = []
    goals_diffs: list[float] = []
    kept_dates: list = []
    refused = missing = 0
    try:
        for row in chosen.itertuples(index=False):
            try:
                built = features.build_match_features(int(row.fixture_id))
            except features.NotEnoughHistoryError:
                # The training table gated on MIN_WINDOW against the raw
                # archive; if Postgres refuses the same fixture, the two
                # histories are not the same depth after all.
                refused += 1
                continue
            if built is None:
                missing += 1
                continue
            served = built.v2_vector[0]
            trained = np.array([getattr(row, c) for c in base_cols], dtype=float)
            diff = np.abs(served - trained)
            diffs.append(diff)
            # Both goals columns, per fixture, so the distribution below is over
            # club-observations rather than over fixtures.
            goals_diffs.extend(diff[-len(GOALS_COLUMNS):])
            kept_dates.extend([row.date] * len(GOALS_COLUMNS))
    finally:
        db.pool.close()

    if refused:
        print(f"  !! {refused} fixtures the training table kept but Postgres "
              f"refused for too little history")
    if missing:
        print(f"  !! {missing} fixtures not in `fixtures` at all")
    if not diffs:
        print("no fixtures could be rebuilt - nothing to compare")
        return 1

    worst = np.max(np.vstack(diffs), axis=0)
    by_column = dict(zip(base_cols, worst))

    form_worst = max(by_column[c] for c in FORM_COLUMNS)
    print(f"  {len(diffs)} fixtures rebuilt through build_match_features()\n")
    print(f"=== 54 form columns ===")
    print(f"  max |served - trained| across all of them: {form_worst:.9f}")
    offenders = [(c, v) for c, v in by_column.items()
                 if c in FORM_COLUMNS and v > TOLERANCE]
    for column, value in sorted(offenders, key=lambda kv: -kv[1])[:10]:
        print(f"    {column:<34} {value:.6f}")
    if not offenders:
        print("  Every form column agrees to float noise. team_form() over a")
        print("  Postgres window and team_form() over the raw archive are the same")
        print("  function of the same history.")

    print(f"\n=== 2 goals columns (expected to differ) ===")
    report_goals(np.array(goals_diffs), pd.Series(kept_dates), df)

    ok = not offenders and not refused and not missing
    print("\n=== verdict ===")
    if ok:
        print("  PASS - the served form row is the trained form row.")
    else:
        print("  FAIL - see the !! lines and offending columns above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
