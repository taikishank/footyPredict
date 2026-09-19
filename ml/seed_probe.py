"""How much of this project's small findings are just the random seed?

Every ablation here compares two *fits* and reports a paired bootstrap over
fixtures. That interval answers "does this fit beat that fit on these rows",
and it is the right interval for that question. It is not an interval for "does
this recipe beat that recipe", because it holds one thing fixed that is not
actually fixed: XGBoost is fitted with `subsample=0.6`, so the row sample per
tree is drawn from `random_state`, and two fits of the *identical* recipe under
two seeds are two different models. The bootstrap resamples fixtures and never
sees that variance component.

This matters because of a specific open lead. FINDINGS_GOALS.md open item 2
and FINDINGS_LEAGUE.md open item 1 record two unrelated noise injections that
both improved v3 significantly - twenty random-partition interactions
(+0.00074 [+0.00034, +0.00113]) and three shuffled Poisson rates (+0.00032
[+0.00003, +0.00062]) - and read them as evidence that v3 is regularised such
that adding noise columns helps it. There is a cheaper explanation. Adding any
column changes how XGBoost consumes its RNG and how it breaks ties between
candidate splits, so a placebo arm is, mechanically, close to a re-seed. If
seed-to-seed spread is of the same order as those two numbers, both are seed
noise and neither is evidence about regularisation.

The probe fits the served v3 recipe under several seeds, changing nothing else,
and asks three things:

1. **How wide is the seed band?** The spread of pooled log loss across seeds is
   an error bar no table in this project currently carries.
2. **How often does a pure re-seed clear the project's own significance bar?**
   Every seed pair is run through the same paired bootstrap the ablations use.
   A recipe compared against itself should never "win"; the rate at which it
   does is the false-positive rate of the bar at this effect size.
3. **Does the published placebo land inside the band?** The shuffled-rate arm
   from FINDINGS_GOALS.md is reproduced exactly and priced against the seed
   distribution rather than against a single fit.

Nothing here is a model change and nothing is written to a served path.

Run:
    ./inference-py/bin/python ml/seed_probe.py --mode split     # ~4 min
    ./inference-py/bin/python ml/seed_probe.py --mode rolling   # ~20 min
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import train_v3 as tv3
import tune
from goals_ablate import shuffled

FEATURES_DIR = Path(__file__).parent / "data" / "features"
GOALS_PATH = FEATURES_DIR / "goals.csv"
OUT_DIR = Path(__file__).parent / "data" / "tune"

#: Eight is enough to put a usable SD on the pooled loss and cheap enough to
#: run under rolling origin. It is not enough to pin down the tail, which is
#: why the pairwise table below reports a rate rather than a p-value.
SEEDS = tuple(range(8))

#: The placebo from FINDINGS_GOALS.md, reproduced column-for-column: the three
#: score-matrix probabilities of the dc_h365_c7 arm, shuffled across fixtures
#: so their marginal distribution survives and every fixture-level association
#: is destroyed.
PLACEBO_COLS = ["dc_h365_c7_p_home", "dc_h365_c7_p_draw", "dc_h365_c7_p_away"]
PLACEBO_SHUFFLE_SEED = 0


def load(mode: str):
    """The v3 table with the placebo's source columns joined on.

    The join is inner and asserted lossless, because a placebo arm scored on
    different rows from the seed arms would not be comparable to either.
    """
    df, cols = tune.load_table("v3")
    goals = pd.read_csv(GOALS_PATH, usecols=["fixture_id"] + PLACEBO_COLS)
    merged = df.merge(goals, on="fixture_id", how="inner")
    assert len(merged) == len(df), (
        f"goals.csv covers {len(merged)} of {len(df)} v3 rows; "
        "the placebo arm would not be paired with the seed arms"
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    y = merged["result"].map(tune.LABEL_MAP).to_numpy()
    fold_list = ra.folds(len(merged), mode)
    val_all = np.concatenate([val for _, val in fold_list])
    return merged, cols, y, fold_list, val_all


def fit(df, cols, y, fold_list, val_all, seed: int,
        recipe: dict | None = None) -> np.ndarray:
    """Per-fixture log loss for one recipe at one seed."""
    params = {**(tune.SERVED_V3 if recipe is None else recipe),
              "random_state": seed}
    probs, _ = tune.evaluate(params, df, cols, y, fold_list)
    return tune.per_match_nll(probs, y[val_all])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("split", "rolling"), default="split")
    ap.add_argument("--recipe", choices=sorted(tv3.RECIPES), default="v2",
                    help="which fit to measure the band of (default: v2, the "
                         "served depth-4 recipe this probe was written for)")
    args = ap.parse_args()

    # The band is a property of the fit, not of the project: a shallower tree
    # has less to vary. Re-measuring it at v3's retuned depth is what makes the
    # re-priced small findings comparable to a bar of the same height.
    recipe = tv3.RECIPES[args.recipe]

    df, cols, y, fold_list, val_all = load(args.mode)
    print(f"{len(cols)} columns, {len(df)} rows, "
          f"{len(val_all)} validation rows over {len(fold_list)} fold(s)")
    print(f"recipe={args.recipe} held fixed at {recipe}\n")

    nll = {}
    for seed in SEEDS:
        nll[seed] = fit(df, cols, y, fold_list, val_all, seed, recipe)
        print(f"  seed {seed}: log loss {nll[seed].mean():.5f}", flush=True)

    losses = np.array([nll[s].mean() for s in SEEDS])
    sd = losses.std(ddof=1)
    print(f"\n1. THE SEED BAND, one recipe, {len(SEEDS)} seeds")
    print(f"   mean {losses.mean():.5f}   SD {sd:.5f}   "
          f"range {losses.min():.5f}..{losses.max():.5f} "
          f"(spread {losses.max() - losses.min():.5f})")

    # Every unordered pair, run through the ablations' own bootstrap. Both
    # members are the same recipe, so every interval that clears zero is a
    # false positive by construction.
    print(f"\n2. A RECIPE AGAINST ITSELF, all {len(SEEDS) * (len(SEEDS) - 1) // 2} seed pairs")
    clears, diffs = 0, []
    for a, b in itertools.combinations(SEEDS, 2):
        boot = ra.paired_bootstrap(nll[a], nll[b])
        diffs.append(abs(boot["improvement"]))
        if boot["ci_low"] > 0 or boot["ci_high"] < 0:
            clears += 1
    n_pairs = len(diffs)
    print(f"   |difference| between two seeds: median {np.median(diffs):.5f}, "
          f"max {max(diffs):.5f}")
    print(f"   pairs whose 95% interval excludes zero: {clears}/{n_pairs} "
          f"({100 * clears / n_pairs:.0f}%)")
    print("   (every one of these is a false positive: same recipe, same rows)")

    placebo_df = shuffled(df, PLACEBO_COLS, seed=PLACEBO_SHUFFLE_SEED)
    placebo_nll = tune.per_match_nll(
        tune.evaluate({**recipe, "random_state": 0},
                      placebo_df, cols + PLACEBO_COLS, y, fold_list)[0],
        y[val_all],
    )
    boot = ra.paired_bootstrap(nll[0], placebo_nll)
    print("\n3. THE PUBLISHED PLACEBO, reproduced")
    print(f"   v3 + 3 shuffled rate columns: {placebo_nll.mean():.5f}")
    print(f"   against v3 at the same seed:  {boot['improvement']:+.5f} "
          f"[{boot['ci_low']:+.5f}, {boot['ci_high']:+.5f}]  "
          f"P(better) = {boot['p_better']:.3f}")
    z = (losses.mean() - placebo_nll.mean()) / sd if sd > 0 else float("nan")
    inside = losses.min() <= placebo_nll.mean() <= losses.max()
    print(f"   against the seed band:        {z:+.2f} SD from the seed mean, "
          f"{'INSIDE' if inside else 'outside'} the observed seed range")

    print("\n4. RE-PRICING THE SMALL FINDINGS against the seed band "
          f"(SD {sd:.5f}, max seed-pair gap {max(diffs):.5f})")
    # (name, effect at the served depth-4 recipe, effect re-priced at v3's
    # depth-2 recipe or None if not re-measured, source). The re-priced column
    # is the point of the whole capacity phase: an effect measured against an
    # over-fitting model is not the effect against the model that ships.
    published = [
        ("shuffled rates placebo", 0.00032, -0.00004, "FINDINGS_GOALS.md"),
        ("random-partition placebo", 0.00074, None, "FINDINGS_LEAGUE.md"),
        ("hfa_lr10", 0.00035, -0.00017, "FINDINGS_VENUE.md"),
        ("xseason_age", 0.00041, 0.00014, "FINDINGS_SCOPE.md"),
        ("g38_s8 window arm", 0.00040, None, "FINDINGS_WINDOW.md"),
        ("league interact arm", 0.00052, None, "FINDINGS_LEAGUE.md"),
        ("dc_rates columns", 0.00098, 0.00101, "FINDINGS_GOALS.md"),
        ("the DC blend", 0.00319, 0.00168, "FINDINGS_GOALS.md"),
        ("atkdef ratings", 0.00763, None, "FINDINGS_RATINGS.md"),
    ]
    print(f"   {'':26s} {'depth 4':>9s} {'depth 2':>9s} {'|SD|':>7s}  "
          f"vs THIS run's band")
    for name, effect, repriced, source in sorted(published, key=lambda r: r[1]):
        # Judge whatever effect belongs to the recipe this run actually fitted,
        # so the bar and the number being held up to it are the same height.
        current = effect if args.recipe == "v2" or repriced is None else repriced
        ratio = abs(current) / sd if sd > 0 else float("nan")
        verdict = ("inside seed noise" if abs(current) <= max(diffs)
                   else "clears the seed band")
        shown = f"{repriced:+.5f}" if repriced is not None else "-"
        print(f"   {name:26s} {effect:+9.5f} {shown:>9s} {ratio:6.1f}  "
              f"{verdict:20s} {source}")
    if args.recipe != "v2":
        print("\n   Judged on the depth-2 column where one exists, against this run's")
        print("   own band. A dash is an arm nobody has re-priced yet.")
    print("\n   NOTE: an arm that adds columns is not only a re-seed - it can carry")
    print("   real information too. This table bounds how much of an effect that")
    print("   size could be fit noise, it does not attribute any particular one.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.recipe == "v2" else f"_{args.recipe}"
    out = OUT_DIR / f"seed_probe_{args.mode}{suffix}.npz"
    np.savez_compressed(
        out,
        fixture_id=df["fixture_id"].to_numpy()[val_all],
        y=y[val_all],
        seeds=np.array(SEEDS),
        seed_nll=np.vstack([nll[s] for s in SEEDS]),
        placebo_nll=placebo_nll,
    )
    print(f"\nper-fixture losses -> {out}")


if __name__ == "__main__":
    main()
