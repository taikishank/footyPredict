"""How long should the form window be, on top of the v3 recipe?

MAX_WINDOW = 8 is an unmeasured constant. This grades windows 4 to 38 against
it on identical rows, identical hyperparameters and the identical rating
columns the ratings phase promoted (atkdef, FINDINGS_RATINGS.md), so the only
thing that moves between arms is how many of a club's past matches its form
vector averages.

Arms, in three groups:

    w4 ... w38        uniform: every form field over the same window.

    g38_s8, g16_s8    split: goals over a long window, match statistics over
                      the incumbent's 8. Phase 9's evidence is specifically
                      about *scoring rate* over 38 matches; possession and
                      card counts have no such claim on a long horizon, and a
                      uniform sweep cannot tell a goals effect from a stats
                      effect.

    w8_nocgr,         the same arms with club_goals_relative dropped. If a
    w38_nocgr         long form window is only rediscovering the 38-match
                      goal column the model already has, w38_nocgr loses far
                      less than w8_nocgr does - that is the redundancy test,
                      and it is the difference between "lengthen the window"
                      and "the horizon is already covered".

Everything else is rating_ablate.py's machinery, imported rather than
restated: the same expanding-window folds, the same paired bootstrap on
per-fixture losses, the same train_v2 hyperparameters. The comparison baseline
here is w8 - the incumbent window - not v2, because the question is whether 8
is wrong, not whether form is worth anything.

The hyperparameters were tuned at window 8 (tune.py:259). A longer window is a
smoother feature and might prefer a different depth or learning rate, so a
null result here is "8 is not wrong *at this tuning*", not "the horizon is
worthless" - see FINDINGS_WINDOW.md.

Run:
    ./inference-py/bin/python ml/build_window_features.py      # ~8 min, once
    ./inference-py/bin/python ml/window_ablate.py --mode split
    ./inference-py/bin/python ml/window_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import train_v2 as tv2
from build_window_features import WINDOWS
from feature_lib import FORM_FIELD_ORDER
from ratings_lib import ARMS

SWEEP_DIR = Path(__file__).parent / "data" / "features" / "window_sweep"
RATINGS_PATH = ra.RATINGS_PATH

GOALS_FIELDS = ("goals_for", "goals_against")
STATS_FIELDS = tuple(f for f in FORM_FIELD_ORDER if f not in GOALS_FIELDS)

CGR = ["home_club_goals_relative", "away_club_goals_relative"]
ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)

BASELINE_ARM = "w8"

# name -> (goals window, stats window, keep club_goals_relative)
ARM_SPECS: dict[str, tuple[int, int, bool]] = {
    **{f"w{w}": (w, w, True) for w in WINDOWS},
    "g38_s8": (38, 8, True),
    "g16_s8": (16, 8, True),
    "w8_nocgr": (8, 8, False),
    "w38_nocgr": (38, 38, False),
}


def form_cols(goals_w: int, stats_w: int) -> list[str]:
    cols = []
    for side in ("home", "away"):
        cols += [f"{side}_form_{f}__w{goals_w}" for f in GOALS_FIELDS]
        cols += [f"{side}_form_{f}__w{stats_w}" for f in STATS_FIELDS]
    return cols


def arm_columns(name: str) -> list[str]:
    goals_w, stats_w, keep_cgr = ARM_SPECS[name]
    return form_cols(goals_w, stats_w) + (CGR if keep_cgr else []) + ATKDEF


def load() -> pd.DataFrame:
    """Every window's form columns, suffixed __w<n>, on one row per fixture.

    Inner joins throughout, so a fixture missing from any input is missing
    from all arms - the pairing the bootstrap relies on.
    """
    # date/result/club_goals_relative come from the window tables themselves
    # rather than 2000_window3_8.csv: that table predates the last archive
    # refresh, and the sweep should score the same fixtures ratings.csv covers.
    df = pd.read_csv(RATINGS_PATH).drop(columns=["date", "league_id"])

    for w in WINDOWS:
        table = pd.read_csv(SWEEP_DIR / f"form_w{w}.csv", parse_dates=["date"])
        form = [c for c in table.columns if c.startswith(("home_form_", "away_form_"))]
        # date, result and club_goals_relative are identical across the tables
        # by construction, so they are taken from the first one only.
        keep = ["fixture_id"] + form + (["date", "result"] + CGR if w == WINDOWS[0] else [])
        table = table[keep].rename(columns={c: f"{c}__w{w}" for c in form})
        df = df.merge(table, on="fixture_id", how="inner")

    df = df.sort_values("date").reset_index(drop=True)
    print(f"{len(df)} fixtures, {len(WINDOWS)} windows, "
          f"{len(arm_columns(BASELINE_ARM))} columns per arm")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    cache = SWEEP_DIR / f"window_losses_{args.mode}.npz"
    df = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    print(f"mode={args.mode}, {len(plan)} fold(s), "
          f"{sum(len(v) for _, v in plan)} validation fixtures total\n")

    results = {}
    if cache.exists():
        stored = np.load(cache)
        results = {name: stored[name] for name in stored.files}
        print(f"reusing cached per-fixture losses from {cache.name}\n")

    for name in ARM_SPECS:
        if name in results:
            continue
        cols = arm_columns(name)
        losses = np.concatenate(
            [ra.fit_and_score(df, cols, y, tr, va) for tr, va in plan]
        )
        results[name] = losses
        print(f"  fitted {name:12s} ({len(cols)} cols)  log loss {losses.mean():.5f}")
        np.savez(cache, **results)

    base = results[BASELINE_ARM]
    print(f"\n{'arm':12s} {'cols':>5s} {'log loss':>9s} {'vs w8':>9s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 70)
    for name in ARM_SPECS:
        if name == BASELINE_ARM:
            print(f"{name:12s} {len(arm_columns(name)):5d} {base.mean():9.5f} "
                  f"{'-':>9s} {'-':>20s} {'-':>10s}")
            continue
        stats = ra.paired_bootstrap(base, results[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:12s} {len(arm_columns(name)):5d} {results[name].mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")

    print("\nPositive 'vs w8' favours the longer/shorter arm over the incumbent "
          "window.\nAn interval spanning zero is a null result regardless of the "
          "point estimate.")

    # What dropping club_goals_relative costs at each end. If the long window
    # has already absorbed the 38-match goal horizon, it should barely notice.
    for window_arm, stripped in (("w8", "w8_nocgr"), ("w38", "w38_nocgr")):
        stats = ra.paired_bootstrap(results[stripped], results[window_arm])
        print(f"\nclub_goals_relative is worth {stats['improvement']:+.5f} at "
              f"{window_arm} "
              f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]")


if __name__ == "__main__":
    main()
