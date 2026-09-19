"""Is a venue split worth anything once the two halves are allowed to share,
and should one match's surprise be bounded before it reaches a rating?

FINDINGS_RWI.md prices an external design against this board and finds two
mechanisms in it that nine phases here have not tested. Both are update rules,
so every arm is a drop-in replacement for the served `atkdef` rating: every arm
carries exactly 63 columns and differs from v3 in the rating rule and nothing
else. That is FINDINGS_VENUE.md's protocol, imported rather than restated so
the folds, the interval and the fit are identical.

    bounded_lin  tanh at a scale where it is the identity. Must reproduce v3
                 exactly; an integrity check on the table, not a hypothesis.
    bounded_t1   the surprise through scale * tanh(err / scale), scale 1
    bounded_t2   ... scale 2
    bounded_t4   ... scale 4
    bounded_c2   the Huber gradient - hard clip at 2 - as a shape cross-check
    couple_c0    VenueAttackDefence unchanged. The second integrity arm: it
                 must reproduce FINDINGS_VENUE.md's -0.00102 `venue` row.
    couple_c10   the two venue slots pulled 10% toward each other per update
    couple_c25   ... 25%
    couple_c50   ... 50%
    couple_c75   ... 75%

Fitted at train_v3.PARAMS by default, not v2's. Every number in
FINDINGS_VENUE.md and FINDINGS_LEAGUE.md was measured at depth 4, and
FINDINGS_CAPACITY.md re-priced the whole board at depth 2 - so a depth-4
measurement is no longer the incumbent, and `--recipe v2` exists only to
reproduce the older rows for comparison.

Three bars, written down in FINDINGS_RWI.md §4 before any of this was run:

1. The couple sweep is read as a *curve*, not as a best arm. Monotone toward
   the coupled end says the venue split never had anything in it and the
   gains are pure variance reduction; an interior optimum says there is a
   venue trait worth carrying at partial strength.
2. No arm under +0.0002 is promoted. FINDINGS_CAPACITY.md §2 measured the
   same-recipe seed band at 0.00007 SD with 11% of seed pairs clearing the
   95% interval, so an interval alone does not settle an effect this size.
3. An optimum on the edge of a sweep extends the sweep before it is read.

Run:
    ./inference-py/bin/python ml/rwi_ablate.py --mode split
    ./inference-py/bin/python ml/rwi_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import train_v2 as tv2
import train_v3 as tv3
import venue_ablate as va
from build_rwi_features import arm_columns
from ratings_lib import ARMS
from rwi_lib import RWI_ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"
RWI_PATH = FEATURES_DIR / "rwi_ratings.csv"

ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
BASELINE = "v3"

#: The couple arms in sweep order, so the curve is read along the axis rather
#: than sorted by the number it produced.
COUPLE_SWEEP = ("couple_c0", "couple_c10", "couple_c25", "couple_c50",
                "couple_c75", "couple_c90", "couple_c100")
COUPLING = {"couple_c0": 0.0, "couple_c10": 0.10, "couple_c25": 0.25,
            "couple_c50": 0.50, "couple_c75": 0.75, "couple_c90": 0.90,
            "couple_c100": 1.00}

#: The bounded sweep, in scale order rather than dict order, for the same
#: reason COUPLE_SWEEP exists.
#:
#: `bounded_lin` is the last entry and not merely the integrity arm: tanh at
#: LINEAR_SCALE *is* the scale -> infinity end of this axis, so the sweep is
#: closed on the right by a point already measured (and equal to v3 exactly).
#: Leaving it out made an interior optimum look like it sat on an edge.
BOUNDED_SWEEP = ("bounded_t05", "bounded_t1", "bounded_t2", "bounded_t4",
                 "bounded_t8", "bounded_lin")

#: FINDINGS_CAPACITY.md §2: the same-recipe seed band, and the floor under
#: anything this phase could promote.
SEED_BAND = 0.0002


def load() -> tuple[pd.DataFrame, list[str]]:
    """The v3 table with every arm's rating columns joined on.

    Inner joins throughout, so a fixture missing from any input is missing
    from all arms - the fixture-for-fixture pairing the bootstrap needs.
    """
    df, form = ra.load()
    rwi = pd.read_csv(RWI_PATH).drop(columns=["date", "league_id"])
    before = len(df)
    df = df.merge(rwi, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, f"{before - len(df)} fixtures lack rwi rating columns"
    return df, form


def arm_plan() -> dict[str, list[str]]:
    """Arm name -> the rating columns it swaps in for atkdef's."""
    plan = {BASELINE: ATKDEF}
    plan.update({name: arm_columns(name) for name in RWI_ARMS})
    return plan


def report_sweep(probs: dict[str, np.ndarray], y: np.ndarray,
                 base: np.ndarray, arms: tuple[str, ...], axis: str) -> None:
    """One sweep, in axis order, with the edge check bar 3 asks for."""
    print(f"\n{'arm':14s} {axis:>9s} {'log loss':>9s} {'vs v3':>9s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 76)
    means = {}
    for name in arms:
        losses = va.loss_of(probs[name], y)
        means[name] = losses.mean()
        stats = ra.paired_bootstrap(base, losses)
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        label = (f"{COUPLING[name]:.2f}" if name in COUPLING
                 else "inf" if name == "bounded_lin"
                 else name.split("_")[-1].lstrip("t").replace("05", "0.5"))
        print(f"{name:14s} {label:>9s} {means[name]:9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")

    best = min(arms, key=lambda n: means[n])
    if best in (arms[0], arms[-1]):
        print(f"\n  NOTE: the optimum ({best}) sits on the edge of this sweep. "
              "Bar 3 says\n  extend the grid before reading it as a result.")
    gain = float(base.mean() - means[best])
    if gain < SEED_BAND:
        print(f"  NOTE: the best arm is {gain:+.5f} over v3, inside the "
              f"{SEED_BAND:.4f} seed band.\n  Bar 2 says it is not promotable "
              "whatever its interval.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    parser.add_argument("--recipe", choices=sorted(tv3.RECIPES), default="v3",
                        help="which fit to measure every arm at (default: v3, "
                             "the recipe FINDINGS_CAPACITY.md re-priced the "
                             "board at)")
    args = parser.parse_args()

    params = tv3.RECIPES[args.recipe]
    suffix = "" if args.recipe == "v3" else f"_{args.recipe}"
    cache = FEATURES_DIR / f"rwi_probs_{args.mode}{suffix}.npz"

    df, form_cols = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    val_idx = np.concatenate([v for _, v in plan])
    y_val = y[val_idx]
    arms = arm_plan()
    print(f"mode={args.mode}, {len(plan)} fold(s), {len(val_idx)} validation "
          f"fixtures, {len(arms)} arms")
    print(f"recipe={args.recipe}: {params}\n")

    probs = {}
    if cache.exists():
        stored = np.load(cache)
        probs = {name: stored[name] for name in stored.files}
        print(f"reusing cached probabilities from {cache.name}\n")

    for name, rating_cols in arms.items():
        if name in probs:
            continue
        cols = form_cols + rating_cols
        probs[name] = np.concatenate(
            [va.fit_probs(df, cols, y, tr, va_idx, params) for tr, va_idx in plan])
        print(f"  fitted {name:12s} ({len(cols)} cols)  "
              f"log loss {va.loss_of(probs[name], y_val).mean():.5f}")
    np.savez(cache, **probs)

    base = va.loss_of(probs[BASELINE], y_val)
    lin = va.loss_of(probs["bounded_lin"], y_val)
    delta = float(np.abs(base - lin).max())
    print(f"\nintegrity: bounded_lin vs v3, max per-fixture |delta| = {delta:.2e}"
          f"  ({'identical' if delta < 1e-9 else 'DIFFERENT - investigate'})")

    print("\n=== does bounding one match's surprise help? ===")
    report_sweep(probs, y_val, base, BOUNDED_SWEEP, "scale")
    stats = ra.paired_bootstrap(va.loss_of(probs["bounded_t2"], y_val),
                                va.loss_of(probs["bounded_c2"], y_val))
    print(f"\n  shape cross-check, clip against tanh at the same scale: "
          f"{stats['improvement']:+.5f}\n  [{stats['ci_low']:+.5f}, "
          f"{stats['ci_high']:+.5f}]. Positive favours the hard clip. Two "
          "shapes\n  disagreeing is a sign the effect is the scale, not the "
          "bounding.")

    print("\n=== is there a venue trait worth carrying at partial strength? ===")
    report_sweep(probs, y_val, base, COUPLE_SWEEP, "coupling")
    print("\n  couple_c0 is VenueAttackDefence, FINDINGS_VENUE.md's -0.00102 arm, "
          "re-priced\n  here at depth 2 like everything else. Read the curve, "
          "not the best row: a\n  shallow interior optimum that never clears v3 "
          "is variance reduction on the\n  way back to one rating, and says the "
          "venue split carried no trait to keep.")

    va.by_outcome(probs, y_val, [BASELINE] + list(RWI_ARMS))
    print("\nFINDINGS_VENUE.md open item 2: away wins are the worst-predicted "
          "outcome by a\nwide margin and no venue parameter has touched it. If "
          "a couple arm pays, this\nis the table that says whether it paid "
          "where the hypothesis predicted.")


if __name__ == "__main__":
    main()
