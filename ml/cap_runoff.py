"""Pick v3's hyperparameters from a flat grid, and test why noise columns help.

ml/tune.py --grid cap2 --mode rolling left the top of the surface flat: ten
configs inside 0.00073 of each other, with the nominal winner (max_depth 1,
2400 rounds) 0.00012 ahead of the second (max_depth 2, 800 rounds). That is
best-of-18 on the rows it is scored on, and the gaps between the leaders are
1-4x the seed-to-seed SD ml/seed_probe.py measured (0.00007). A ranking cannot
settle it; a pairwise comparison can.

Two jobs, and the second is the more interesting one.

**The runoff.** Every leading config is fitted again and paired against the
served recipe and against each other on identical fixtures, which is the
comparison FINDINGS_RATINGS.md had to make when five arms shared a baseline and
their intervals all overlapped. `cap2` cached only its own winner's
probabilities, so the candidates are re-fitted here rather than replayed.

**The placebo, as a function of depth.** FINDINGS_GOALS.md open item 2 and
FINDINGS_LEAGUE.md open item 1 record two noise injections that significantly
improved v3, and read them as v3 being unable to reach the interactions it
wants. `cap` killed that reading: depth is monotone the other way, so v3 is
over-fitting at the served depth 4, not under-fitting. ml/seed_probe.py killed
the other cheap explanation: the seed band is 0.00007 SD and the placebo sits
3.5 SD outside it, so it is not fit noise either.

What is left is the same evidence with the sign flipped. If a model is
over-capacity, columns carrying no information are not merely harmless - they
absorb splits that would otherwise have gone to real columns, which is what
colsample_bytree does deliberately. Under that reading the placebo gain is an
*implicit regularisation* effect, and it should shrink as the tree shrinks,
reaching zero once the depth is right. Re-running the published placebo at each
candidate depth tests exactly that, and it is the only explanation left standing.

Run:
    ./inference-py/bin/python ml/cap_runoff.py --mode rolling   # ~12 min
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

import rating_ablate as ra
import tune
from goals_ablate import shuffled
from seed_probe import PLACEBO_COLS, PLACEBO_SHUFFLE_SEED, load

OUT_DIR = Path(__file__).parent / "data" / "tune"

#: The served recipe plus every config within 0.0006 of cap2's rolling winner,
#: named by what distinguishes it. `d1` needs 2400 rounds to reach the same
#: capacity 800 rounds give `d2`, which is the point of carrying both.
CANDIDATES = {
    "served": {"max_depth": 4, "reg_lambda": 100, "n_estimators": 800},
    "d3": {"max_depth": 3, "reg_lambda": 30, "n_estimators": 800},
    "d2": {"max_depth": 2, "reg_lambda": 30, "n_estimators": 800},
    "d2_l10": {"max_depth": 2, "reg_lambda": 10, "n_estimators": 800},
    "d1": {"max_depth": 1, "reg_lambda": 100, "n_estimators": 2400},
}
REFERENCE = "served"


def config(name: str) -> dict:
    return {**tune.SERVED_V3, **CANDIDATES[name], "random_state": 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("split", "rolling"), default="rolling")
    args = ap.parse_args()

    df, cols, y, fold_list, val_all = load(args.mode)
    y_val = y[val_all]
    placebo_df = shuffled(df, PLACEBO_COLS, seed=PLACEBO_SHUFFLE_SEED)
    print(f"{len(cols)} columns, {len(val_all)} validation rows, "
          f"{len(CANDIDATES)} candidates x 2 arms\n")

    nll, placebo_nll, probs_of = {}, {}, {}
    for name in CANDIDATES:
        probs, _ = tune.evaluate(config(name), df, cols, y, fold_list)
        nll[name] = tune.per_match_nll(probs, y_val)
        probs_of[name] = probs

        p_probs, _ = tune.evaluate(config(name), placebo_df,
                                   cols + PLACEBO_COLS, y, fold_list)
        placebo_nll[name] = tune.per_match_nll(p_probs, y_val)
        print(f"  {name:8s} {nll[name].mean():.5f}   "
              f"+placebo {placebo_nll[name].mean():.5f}", flush=True)

    print(f"\n1. THE RUNOFF, every candidate against '{REFERENCE}'")
    print(f"   {'arm':10s} {'log loss':>9s} {'vs ref':>9s}  {'95% CI':>20s}  P")
    for name in CANDIDATES:
        if name == REFERENCE:
            print(f"   {name:10s} {nll[name].mean():9.5f} {'-':>9s}")
            continue
        b = ra.paired_bootstrap(nll[REFERENCE], nll[name])
        print(f"   {name:10s} {nll[name].mean():9.5f} {b['improvement']:+9.5f}  "
              f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  {b['p_better']:.3f}")

    print("\n2. THE LEADERS AGAINST EACH OTHER (positive favours the first)")
    leaders = [n for n in CANDIDATES if n != REFERENCE]
    for a, b_name in itertools.combinations(leaders, 2):
        b = ra.paired_bootstrap(nll[b_name], nll[a])
        sep = "separated" if (b["ci_low"] > 0 or b["ci_high"] < 0) else "tied"
        print(f"   {a:8s} vs {b_name:8s} {b['improvement']:+.5f}  "
              f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  "
              f"P={b['p_better']:.3f}  {sep}")

    print("\n3. THE PLACEBO AS A FUNCTION OF DEPTH")
    print("   If noise columns help by absorbing splits an over-capacity tree")
    print("   would otherwise waste, the gain shrinks toward zero as depth drops.")
    print(f"   {'arm':10s} {'depth':>5s} {'placebo gain':>13s}  {'95% CI':>20s}  P")
    for name in CANDIDATES:
        b = ra.paired_bootstrap(nll[name], placebo_nll[name])
        print(f"   {name:10s} {CANDIDATES[name]['max_depth']:5d} "
              f"{b['improvement']:+13.5f}  "
              f"[{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  {b['p_better']:.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"cap_runoff_{args.mode}.npz"
    np.savez_compressed(
        out,
        fixture_id=df["fixture_id"].to_numpy()[val_all],
        y=y_val,
        names=np.array(list(CANDIDATES)),
        nll=np.vstack([nll[n] for n in CANDIDATES]),
        placebo_nll=np.vstack([placebo_nll[n] for n in CANDIDATES]),
        **{f"probs_{n}": probs_of[n] for n in CANDIDATES},
    )
    print(f"\nper-fixture losses and probabilities -> {out}")


if __name__ == "__main__":
    main()
