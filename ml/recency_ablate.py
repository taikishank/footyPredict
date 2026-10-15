"""Does the model need to forget?

FINDINGS_LEAGUE.md open item 4: all five leagues' outcome rates drift across
eras by more than the leagues differ from each other (a COVID-era home-win
dip moves all five together, 2019-12..2022-02), and nothing in the model is
allowed to know what year it is - on purpose, Phase 9's career-appearance bug
(FINDINGS_PHASE9.md) was exactly a column whose *value* tracked the archive's
own maturation rather than football, and it scored well on a chronological
split for entirely the wrong reason.

Row weighting is a different mechanism and does not reintroduce that bug: no
arm here ever adds a date/era column to `cols`. The feature set is v3's own
63 columns, fixed across every arm; only `sample_weight` varies, which
changes how much each row counts toward the fit and hands the model nothing
it can split on to recover "what year is it." Dixon-Coles already does this
for the goals model (goals_lib.py's half-life decay); this phase asks whether
the same mechanism is worth anything to the classifier.

Arms:

    baseline     every row weighted 1. The served recipe exactly.
    hl180        0.5 ** (days_before_cutoff / 180)
    hl365        ... / 365 - Dixon-Coles' own fitted scale
    hl730        ... / 730
    hl1460       ... / 1460 - the bracket is widened past goals_lib's own
                 180/365/730 because that sweep was calibrated for a single
                 league's single-season fit; this one pools 26 years across
                 five leagues, so the half-life that matters here is not
                 known to sit in the same place.
    shuffle_365  hl365's weights, permuted within league and within each
                 fold's training rows. Same histogram, no link to which row
                 is actually recent - the placebo. goals_ablate.shuffled()
                 is the precedent: a column permuted within league keeps the
                 league's own level and loses only the link to the fixture.
                 FINDINGS_LEAGUE.md's own `interact` arm is the warning this
                 control exists for - it cleared its interval until a
                 same-shape placebo beat it, proving the gain was a
                 tree-depth artifact rather than a football one.
    anchor_365   0.5 ** (|date - random_anchor| / 365), anchor drawn once per
                 fold from that fold's own training date range. Keeps the
                 exponential *shape* but decouples it from "closest to now" -
                 separates "recency specifically" from "any temporally-local
                 reweighting helps," if hl365 clears shuffle_365's bar.

Per fold, the weight is computed relative to that fold's own cutoff, not a
single global "today": `asof_date = df["date"].iloc[val_idx].min()`, and
`days_before_cutoff = (asof_date - date).days` for every training row -
never looks at val_idx's own dates beyond that one scalar. A fold's training
weights are asserted to come entirely from before its own asof_date.

Fitted at train_v3.PARAMS by default, not v2's. FINDINGS_LEAGUE.md's
`interact`-vs-`placebo` result is a depth-4-specific failure mode
(FINDINGS_CAPACITY.md re-priced the whole board at depth 2), and uneven row
weighting carries the same two-sided risk: over-capacity, a shuffled weight
can buy variance reduction unrelated to recency; under-capacity, a genuine
recency effect can get muted by leaves too coarse to feel it. `--recipe v2`
exists to compare.

Two modes, rating_ablate.folds()'s: `split` is one chronological 80/20 cut
that happens to land at 2022-02-05, inside the COVID-era dip this phase is
about - do not read a split-only result as the answer, the same caution
rating_ablate.py itself gives. `rolling`'s five expanding-window folds have
asof dates that straddle that dip unevenly too, which is why this phase
prints a per-fold breakdown and not only the pooled table.

Run:
    ./inference-py/bin/python ml/recency_ablate.py --mode split
    ./inference-py/bin/python ml/recency_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import rating_ablate as ra
import train_v2 as tv2
import train_v3 as tv3
from ratings_lib import ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"

ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
BASELINE = "v3"

HALF_LIVES = {"hl180": 180.0, "hl365": 365.0, "hl730": 730.0, "hl1460": 1460.0}
SHUFFLE_ARM = "shuffle_365"
ANCHOR_ARM = "anchor_365"
SHUFFLE_SEED = 0
ANCHOR_SEED = 0

ARMS_ORDER = [BASELINE] + list(HALF_LIVES) + [SHUFFLE_ARM, ANCHOR_ARM]


def load() -> tuple[pd.DataFrame, list[str]]:
    """The v3 table. No new merge - the weight is computed from df["date"],
    already present."""
    df, baseline = ra.load()
    return df, baseline + ATKDEF


def recency_weight(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
                   half_life: float) -> np.ndarray:
    """0.5 ** (days before this fold's asof date / half_life), for train rows.

    asof_date is the earliest date in val_idx - the moment this fold starts
    predicting - never a date later than anything in val_idx, and never
    informed by anything about val_idx except that one scalar.
    """
    dates = df["date"]
    asof_date = dates.iloc[val_idx].min()
    train_dates = dates.iloc[train_idx]
    assert (train_dates <= asof_date).all(), \
        "a training row is after this fold's validation window"
    days_before = (asof_date - train_dates).dt.days.to_numpy().astype(float)
    return 0.5 ** (days_before / half_life)


def shuffled_weight(weight: np.ndarray, league: np.ndarray, seed: int) -> np.ndarray:
    """`weight` permuted within each league, so the histogram survives but the
    link to which row is actually recent does not. Mirrors
    goals_ablate.shuffled()'s within-league construction."""
    rng = np.random.default_rng(seed)
    out = weight.copy()
    for value in np.unique(league):
        rows = np.flatnonzero(league == value)
        out[rows] = weight[rows[rng.permutation(len(rows))]]
    return out


def anchor_weight(df: pd.DataFrame, train_idx: np.ndarray, half_life: float,
                  seed: int) -> np.ndarray:
    """0.5 ** (|date - random anchor| / half_life), anchor drawn once per fold
    from train_idx's own date range - keeps the exponential's shape (still
    temporally local) but severs "closest to now" from the weighting."""
    dates = df["date"].iloc[train_idx]
    rng = np.random.default_rng(seed)
    lo, hi = dates.min(), dates.max()
    offset_days = rng.integers(0, max(int((hi - lo).days), 1) + 1)
    anchor = lo + pd.Timedelta(int(offset_days), unit="D")
    distance = (dates - anchor).abs().dt.days.to_numpy().astype(float)
    return 0.5 ** (distance / half_life)


def effective_sample_size(weight: np.ndarray) -> float:
    """Kish's ESS: (sum w)^2 / sum(w^2). How many uniformly-weighted rows this
    weighting is worth, in the units len(train_idx) is already printed in."""
    return float(weight.sum() ** 2 / (weight ** 2).sum())


def fit_and_score_weighted(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                           train_idx: np.ndarray, val_idx: np.ndarray,
                           weight: np.ndarray, params: dict) -> np.ndarray:
    """ra.fit_and_score's shape, with a sample_weight on the training rows
    only - evaluation on val_idx stays unweighted so every arm's log loss is
    comparable."""
    model = XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", **params,
    )
    model.fit(df[cols].iloc[train_idx], y[train_idx], sample_weight=weight)
    probs = model.predict_proba(df[cols].iloc[val_idx])
    return -np.log(np.clip(probs[np.arange(len(val_idx)), y[val_idx]], 1e-15, 1.0))


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
    cache = FEATURES_DIR / f"recency_losses_{args.mode}{suffix}.npz"

    df, cols = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    league = df["league_id"].to_numpy()
    plan = ra.folds(len(df), args.mode)
    val_lengths = [len(va) for _, va in plan]
    print(f"mode={args.mode}, {len(plan)} fold(s), {sum(val_lengths)} "
          f"validation fixtures total")
    print(f"recipe={args.recipe}: {params}\n")

    results = {}
    if cache.exists():
        stored = np.load(cache)
        results = {name: stored[name] for name in stored.files}
        print(f"reusing cached per-fixture losses from {cache.name}\n")

    ess_lines = []
    for name in ARMS_ORDER:
        if name in results:
            continue
        losses = []
        for tr, va in plan:
            if name == BASELINE:
                losses.append(ra.fit_and_score(df, cols, y, tr, va, params=params))
                continue
            if name in HALF_LIVES:
                weight = recency_weight(df, tr, va, HALF_LIVES[name])
            elif name == SHUFFLE_ARM:
                weight = recency_weight(df, tr, va, HALF_LIVES["hl365"])
                weight = shuffled_weight(weight, league[tr], SHUFFLE_SEED)
            elif name == ANCHOR_ARM:
                weight = anchor_weight(df, tr, HALF_LIVES["hl365"], ANCHOR_SEED)
            else:
                raise KeyError(name)
            ess_lines.append(f"    {name:12s} fold n_train={len(tr):6d}  "
                              f"ESS={effective_sample_size(weight):9.1f}")
            losses.append(
                fit_and_score_weighted(df, cols, y, tr, va, weight, params))
        results[name] = np.concatenate(losses)
        print(f"  fitted {name:12s} ({len(cols)} cols)  "
              f"log loss {results[name].mean():.5f}")
        np.savez(cache, **results)

    if ess_lines:
        print("\neffective sample size (Kish's, (sum w)^2 / sum w^2) per fold:")
        print("\n".join(ess_lines))

    base = results[BASELINE]
    print(f"\n{'arm':12s} {'log loss':>9s} {'vs v3':>9s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 64)
    for name in ARMS_ORDER:
        if name == BASELINE:
            print(f"{name:12s} {base.mean():9.5f} {'-':>9s} {'-':>20s} {'-':>10s}")
            continue
        stats = ra.paired_bootstrap(base, results[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:12s} {results[name].mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")

    print("\nPositive 'vs v3' favours the arm. An interval spanning zero is a "
          "null result\nregardless of where the point estimate sits. The "
          "promotion bar is a half-life arm\nclearing zero AND beating "
          f"{SHUFFLE_ARM}'s interval - the same elimination\nstructure that "
          "caught FINDINGS_LEAGUE.md's `interact` being a placebo artifact.")

    best_hl = min(HALF_LIVES, key=lambda n: results[n].mean())
    stats = ra.paired_bootstrap(results[SHUFFLE_ARM], results[best_hl])
    print(f"\n{best_hl} over {SHUFFLE_ARM}: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]\n"
          "A null here means the weighting helps for a reason that is not "
          "recency.")

    val_idx = np.concatenate([va for _, va in plan])
    val_league = league[val_idx]
    print(f"\nvs v3, per league:\n\n{'arm':12s}" +
          "".join(f"{value:>10d}" for value in np.unique(val_league)))
    print("-" * (12 + 10 * len(np.unique(val_league))))
    for name in ARMS_ORDER[1:]:
        row = "".join(
            f"{(base[val_league == value] - results[name][val_league == value]).mean():>+10.5f}"
            for value in np.unique(val_league)
        )
        print(f"{name:12s}{row}")

    if args.mode == "rolling":
        print(f"\nvs v3, per fold (fold asof dates straddle the COVID-era dip "
              f"unevenly,\nso a pooled number can hide a fold-specific effect):"
              f"\n\n{'arm':12s}" +
              "".join(f"{i:>10d}" for i in range(len(plan))))
        print("-" * (12 + 10 * len(plan)))
        bounds = np.cumsum([0] + val_lengths)
        for name in ARMS_ORDER[1:]:
            row = "".join(
                f"{(base[bounds[i]:bounds[i+1]] - results[name][bounds[i]:bounds[i+1]]).mean():>+10.5f}"
                for i in range(len(plan))
            )
            print(f"{name:12s}{row}")


if __name__ == "__main__":
    main()
