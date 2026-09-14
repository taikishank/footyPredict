"""Is a score model worth anything to the served classifier, and in what form?

Runs only if `build_goals_features.py --diagnose` cleared its pre-registered
gate; that file's docstring holds the gate, and `FINDINGS_GOALS.md` holds the
result.

The diagnose measures the score model *standalone*. This measures what it is
worth to v3, which is a different question with three possible answers, and
the arms are chosen to tell them apart:

    dc_probs    v3 + the three probabilities the score matrix implies. The
                real arm. A tree splits one column at a time and cannot
                construct a Poisson convolution from two rates, so this is
                the one transform it genuinely cannot reach by itself.
    dc_rates    v3 + the two fitted rates only. The control that decides what
                a positive `dc_probs` means: if the rates alone buy the same
                thing, the gain is a better rating and not the score matrix,
                and the honest write-up is a ratings finding.
    dc_full     both, for completeness.
    dc_swap     the form columns + the score model, with `atkdef`'s seven
                rating columns dropped. Asks whether the two are substitutes -
                the batch fit and the online one estimate the same quantity.
    placebo     three probability columns from *shuffled* rates, so the arm
                has the same shape and none of the information.
                FINDINGS_VENUE.md is why: a four-column split of the same
                shape scored positively there until the placebo beat it.
    blend       not a column set at all - v3's own probabilities combined
                with the score model's in log space, the weight fitted on a
                held-out tail of each fold's training slice. The classical
                way two forecasters are combined, and the only arm that does
                not ask a tree to learn the combination from 19,797 rows.

Machinery is rating_ablate.py's, imported rather than restated: same
expanding-window folds, same train_v2 hyperparameters, same paired bootstrap.
The cache holds probabilities rather than losses (FINDINGS_MARKET.md open item
5), so the by-outcome table - the one that made this phase's hypothesis - is
free.

Run:
    ./inference-py/bin/python ml/goals_ablate.py --mode split
    ./inference-py/bin/python ml/goals_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import train_v2 as tv2
import venue_ablate as vabl
from build_goals_features import FIELDS, OUT_PATH, arm_columns, arm_name
from ratings_lib import ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"
ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
BASELINE = "v3"
OUTCOMES = ("home_win", "draw", "away_win")

#: Fraction of each fold's training slice held out, chronologically, to fit
#: the blend weight on. Small: the weight is one number, and every row spent
#: here is a row the arm's own model does not get.
BLEND_HOLDOUT = 0.15


def load(half_life: float, cadence: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The v3 table with one goals arm's columns joined on."""
    df, form = ra.load()
    goals = pd.read_csv(OUT_PATH).drop(columns=["date", "league_id"])
    keep = ["fixture_id"] + arm_columns(half_life, cadence)
    before = len(df)
    df = df.merge(goals[keep], on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, f"{before - len(df)} fixtures lack goals columns"
    return df, form, arm_columns(half_life, cadence)


def blend_probs(v3: np.ndarray, dc: np.ndarray, weight: float) -> np.ndarray:
    """Log-linear pool of two forecasters, renormalised.

    Log space rather than a linear mixture because both inputs are already
    probabilities a log loss will be taken of: a linear pool cannot be sharper
    than its sharpest input, and FINDINGS_MARKET.md's finding is that v3 is
    not sharp enough. A log pool can be.
    """
    stacked = ((1.0 - weight) * np.log(np.clip(v3, 1e-15, 1.0))
               + weight * np.log(np.clip(dc, 1e-15, 1.0)))
    stacked = np.exp(stacked - stacked.max(axis=1, keepdims=True))
    return stacked / stacked.sum(axis=1, keepdims=True)


def fit_blend_weight(v3: np.ndarray, dc: np.ndarray, y: np.ndarray) -> float:
    """The mixing weight, by grid search on a held-out slice.

    A grid rather than an optimiser because there is exactly one parameter on
    [0, 1] and the loss is smooth in it - and because the *shape* of that
    curve is worth printing.
    """
    grid = np.linspace(0.0, 1.0, 41)
    losses = [loss_of(blend_probs(v3, dc, w), y).mean() for w in grid]
    return float(grid[int(np.argmin(losses))])


def loss_of(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def shuffled(df: pd.DataFrame, columns: list[str], seed: int = 0) -> pd.DataFrame:
    """The placebo: every arm column permuted within its league.

    Within a league rather than globally, so the placebo keeps the marginal
    distribution and the league's scoring level and loses only the link to the
    fixture - the same construction FINDINGS_VENUE.md's random venue split used.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for _, rows in df.groupby("league_id").groups.items():
        order = rng.permutation(len(rows))
        for column in columns:
            out.loc[rows, column] = df.loc[rows, column].to_numpy()[order]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    parser.add_argument("--half-life", type=float, default=365.0)
    parser.add_argument("--cadence", type=float, default=30.0)
    args = parser.parse_args()

    arm = arm_name(args.half_life, args.cadence)
    cache = FEATURES_DIR / f"goals_probs_{args.mode}_{arm}.npz"

    df, form_cols, goals_cols = load(args.half_life, args.cadence)
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    val_idx = np.concatenate([v for _, v in plan])
    y_val = y[val_idx]

    rate_cols = goals_cols[:2]
    prob_cols = goals_cols[2:]
    v3_cols = form_cols + ATKDEF

    arms = {
        BASELINE: v3_cols,
        "dc_probs": v3_cols + prob_cols,
        "dc_rates": v3_cols + rate_cols,
        "dc_full": v3_cols + goals_cols,
        "dc_swap": form_cols + goals_cols,
        "placebo": v3_cols + prob_cols,
    }
    print(f"mode={args.mode}, arm={arm}, {len(plan)} fold(s), "
          f"{len(val_idx)} validation fixtures, {len(arms) + 1} arms\n")

    probs = {}
    if cache.exists():
        stored = np.load(cache)
        probs = {name: stored[name] for name in stored.files}
        print(f"reusing cached probabilities from {cache.name}\n")

    placebo_df = None
    for name, cols in arms.items():
        if name in probs:
            continue
        source = df
        if name == "placebo":
            placebo_df = shuffled(df, prob_cols) if placebo_df is None else placebo_df
            source = placebo_df
        probs[name] = np.concatenate(
            [vabl.fit_probs(source, cols, y, tr, va) for tr, va in plan])
        print(f"  fitted {name:10s} ({len(cols)} cols)  "
              f"log loss {loss_of(probs[name], y_val).mean():.5f}")

    # The blend is fitted per fold on a chronological tail of that fold's
    # training slice, so the weight never sees the rows it is scored on.
    if "blend" not in probs:
        dc_all = df[prob_cols].to_numpy()
        pooled, weights = [], []
        for train_idx, fold_val in plan:
            cut = int(len(train_idx) * (1.0 - BLEND_HOLDOUT))
            inner_train, inner_val = train_idx[:cut], train_idx[cut:]
            inner_probs = vabl.fit_probs(df, v3_cols, y, inner_train, inner_val)
            weight = fit_blend_weight(inner_probs, dc_all[inner_val], y[inner_val])
            weights.append(weight)
            fold_v3 = vabl.fit_probs(df, v3_cols, y, train_idx, fold_val)
            pooled.append(blend_probs(fold_v3, dc_all[fold_val], weight))
        probs["blend"] = np.concatenate(pooled)
        print(f"  fitted {'blend':10s} (weights per fold: "
              f"{', '.join(f'{w:.2f}' for w in weights)})  "
              f"log loss {loss_of(probs['blend'], y_val).mean():.5f}")
    np.savez(cache, **probs)

    base = loss_of(probs[BASELINE], y_val)
    order = [n for n in probs if n != BASELINE]
    print(f"\n{'arm':12s} {'log loss':>9s} {'vs v3':>9s} {'95% CI':>20s} "
          f"{'P(better)':>10s}")
    print("-" * 64)
    print(f"{BASELINE:12s} {base.mean():9.5f} {'-':>9s} {'-':>20s} {'-':>10s}")
    for name in order:
        stats = ra.paired_bootstrap(base, loss_of(probs[name], y_val))
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:12s} {loss_of(probs[name], y_val).mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")
    print("\nPositive 'vs v3' favours the arm. An interval spanning zero is a null "
          "result\nregardless of where the point estimate sits.")

    # The two comparisons that decide what a positive dc_probs would mean.
    stats = ra.paired_bootstrap(loss_of(probs["placebo"], y_val),
                                loss_of(probs["dc_probs"], y_val))
    print(f"\ndc_probs against its placebo: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  "
          f"P(real better) = {stats['p_better']:.3f}")
    stats = ra.paired_bootstrap(loss_of(probs["dc_rates"], y_val),
                                loss_of(probs["dc_probs"], y_val))
    print(f"dc_probs against dc_rates:   {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  "
          f"P(matrix better) = {stats['p_better']:.3f}")
    print("The second is the one that says whether the *score matrix* earned "
          "anything,\nor whether a better pair of rates would have done as well.")

    vabl.by_outcome(probs, y_val, [BASELINE] + order)


if __name__ == "__main__":
    main()
