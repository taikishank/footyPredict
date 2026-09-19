"""Is an opponent-strength rating worth adding to the served model, and which one?

The incumbent's team_form() is opponent-blind - eight matches against
relegation sides and eight against the top four produce the same vector. This
grades the three candidate fixes in ratings_lib against the *served* v2
recipe, on identical rows and an identical split, differing only in which
rating columns they get.

Arms:
    elo        plain W/D/L Elo
    gdelo      Elo with the margin of victory in the update
    atkdef     separate attack/defence ratings, Poisson-fitted
    atkdef_d   the same, expectations only - the four level columns dropped
               because they still correlate +0.17 with the fixture date
               (build_rating_features.py --diagnose), and Phase 9's career
               appearance bug is the precedent for taking that seriously
    all        every arm at once, to see whether they are substitutes

Two modes, because they answer different questions:

    --mode split    one chronological 80/20 split. Directly comparable with
                    every number in FINDINGS_PHASE9.md and with train_v2.py's
                    logged log loss, and fast.

    --mode rolling  expanding-window rolling origin: train on everything
                    before a cut, score the next slice, repeat. The honest
                    mode, and the one that matters here. Every hyperparameter
                    in train_v2.py was selected on the single split above -
                    tune.py:259 says so itself - and testing five more arms on
                    that same 7,919 rows would compound the problem: the best
                    of five arms on one validation set beats the baseline by
                    some margin even when all five are worthless.

The verdict is a paired bootstrap on per-fixture losses, not a log loss
ranking, matching the promotion bar in PLAYER_STATS_SPEC.md §6: both arms
score the same fixtures, so resampling the *difference* lets a chaotic week
inflate both losses together and cancel.

Run:
    ./inference-py/bin/python ml/rating_ablate.py --mode split
    ./inference-py/bin/python ml/rating_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

import train_v2 as tv2
from ratings_lib import ARMS

FEATURES_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"
RATINGS_PATH = Path(__file__).parent / "data" / "features" / "ratings.csv"

BOOTSTRAP_SAMPLES = 2000
RNG_SEED = 0

# Expanding-window folds. Five cuts over the back half of the archive gives
# each fold ~2,600 validation fixtures - small enough that a single fold's
# interval is wide, which is the point: pooling the per-fixture losses across
# folds is what produces an interval worth reading.
N_FOLDS = 5

_ELO = list(ARMS["elo"].FIELD_ORDER)
_GDELO = list(ARMS["gdelo"].FIELD_ORDER)
_ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
_ATKDEF_EXPECTATIONS = ["exp_goals_home", "exp_goals_away", "exp_goal_diff"]

ARM_COLUMNS = {
    "elo": _ELO,
    "gdelo": _GDELO,
    "atkdef": _ATKDEF,
    "atkdef_d": _ATKDEF_EXPECTATIONS,
    "all": _ELO + _GDELO + _ATKDEF,
}


def load() -> tuple[pd.DataFrame, list[str]]:
    """The v2 feature table with the rating columns joined on.

    An inner join on fixture_id, so the rows are exactly the ones the served
    model trains on today: build_features.py's MIN_WINDOW gate drops a
    fixture until both clubs have three matches of history, while the rating
    replay emits a row for every fixture in the archive.
    """
    form = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    ratings = pd.read_csv(RATINGS_PATH, parse_dates=["date"]).drop(columns=["date"])
    df = form.merge(ratings, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)

    form_cols = [c for c in df.columns if c.startswith(("home_form_", "away_form_"))]
    baseline = form_cols + ["home_club_goals_relative", "away_club_goals_relative"]
    print(f"{len(df)} rows ({len(form)} in the v2 table, "
          f"{len(form) - len(df)} without a rating row)")
    return df, baseline


#: The recipe every finding in this project was measured at, imported from
#: train_v2 rather than restated so an ablation measures its columns and not a
#: differently-tuned model - the exact confound train_v2.py's own docstring
#: records having introduced against v1.
#:
#: It is a dict rather than six keyword arguments because it is no longer the
#: only recipe. ml/tune.py --grid cap2 retuned v3's capacity and ml/train_v3.py
#: carries the result (max_depth 2, reg_lambda 30), which is a different model
#: from this one - so every number in FINDINGS_RATINGS.md onward is now a
#: measurement *at depth 4*, and re-pricing one of them means passing a
#: different `params` here rather than editing this constant.
#:
#: Note the direction: train_v3 imports this module, so this module must never
#: import train_v3. A caller that wants v3's capacity passes train_v3.PARAMS in.
V2_PARAMS = {
    "n_estimators": tv2.N_ESTIMATORS,
    "max_depth": tv2.MAX_DEPTH,
    "learning_rate": tv2.LEARNING_RATE,
    "reg_lambda": tv2.REG_LAMBDA,
    "subsample": tv2.SUBSAMPLE,
    "random_state": tv2.RANDOM_STATE,
}


def fit_and_score(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                  train_idx: np.ndarray, val_idx: np.ndarray,
                  params: dict | None = None) -> np.ndarray:
    """Per-fixture log loss on the validation slice.

    `params` defaults to V2_PARAMS, so a caller that does not pass it gets the
    fit every published finding was measured with and reproduces it exactly.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        **(V2_PARAMS if params is None else params),
    )
    model.fit(df[cols].iloc[train_idx], y[train_idx])
    probs = model.predict_proba(df[cols].iloc[val_idx])
    return -np.log(np.clip(probs[np.arange(len(val_idx)), y[val_idx]], 1e-15, 1.0))


def paired_bootstrap(baseline_losses: np.ndarray, arm_losses: np.ndarray) -> dict[str, float]:
    """Positive `improvement` favours the arm. See ablate.py's twin."""
    diff = baseline_losses - arm_losses
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "improvement": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_better": float((means > 0).mean()),
    }


def folds(n_rows: int, mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if mode == "split":
        cut = int(n_rows * (1 - tv2.VAL_FRACTION))
        return [(np.arange(cut), np.arange(cut, n_rows))]

    # Expanding window: the first cut sits at half the archive, and each fold
    # validates on the slice up to the next cut while training on everything
    # before it. Never trains on anything later than it scores.
    cuts = np.linspace(n_rows // 2, n_rows, N_FOLDS + 1).astype(int)
    return [(np.arange(cuts[i]), np.arange(cuts[i], cuts[i + 1])) for i in range(N_FOLDS)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    # Per-fixture losses are cached per mode so the analysis below can be
    # re-read or extended without refitting - the rolling mode is 30 fits and
    # ~17 minutes, which is too expensive to pay again for a follow-up
    # question about numbers already computed.
    cache = Path(__file__).parent / "data" / "features" / f"rating_losses_{args.mode}.npz"

    df, baseline_cols = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = folds(len(df), args.mode)
    print(f"mode={args.mode}, {len(plan)} fold(s), "
          f"{sum(len(v) for _, v in plan)} validation fixtures total\n")

    if cache.exists():
        stored = np.load(cache)
        results = {name: stored[name] for name in stored.files}
        print(f"reusing cached per-fixture losses from {cache.name}\n")
    else:
        results = {}

    for name, extra in [("baseline (v2)", [])] + list(ARM_COLUMNS.items()):
        if name in results:
            continue
        cols = baseline_cols + extra
        # Pooled across folds so the bootstrap resamples every scored fixture,
        # not a per-fold mean. Each fold is a different model; the losses are
        # still matched fixture-for-fixture between arms, which is all the
        # pairing needs.
        losses = np.concatenate([fit_and_score(df, cols, y, tr, va) for tr, va in plan])
        results[name] = losses
        print(f"  fitted {name:16s} ({len(cols)} cols)  log loss {losses.mean():.5f}")
    np.savez(cache, **results)

    base = results["baseline (v2)"]
    print(f"\n{'arm':16s} {'cols':>5s} {'log loss':>9s} {'vs v2':>9s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 74)
    print(f"{'baseline (v2)':16s} {len(baseline_cols):5d} {base.mean():9.5f} "
          f"{'-':>9s} {'-':>20s} {'-':>10s}")
    for name, extra in ARM_COLUMNS.items():
        stats = paired_bootstrap(base, results[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:16s} {len(baseline_cols) + len(extra):5d} "
              f"{results[name].mean():9.5f} {stats['improvement']:+9.5f} "
              f"{ci:>20s} {stats['p_better']:10.3f}")

    print("\nPositive 'vs v2' favours the arm. An interval spanning zero is a "
          "null result\nregardless of where the point estimate sits.")

    # Each row above is an arm against the *baseline*, and two such intervals
    # overlapping says nothing about whether the arms differ from each other -
    # their errors are highly correlated, since they score the same fixtures
    # with the same 56 shared columns. The question "is the best arm actually
    # better than the simplest one" needs the arms paired directly, which is
    # what this does.
    best = min(ARM_COLUMNS, key=lambda name: results[name].mean())
    print(f"\nbest arm by log loss: {best}. Paired directly against each other arm:")
    print(f"\n{'arm':16s} {'vs ' + best:>12s} {'95% CI':>20s} {'P(best wins)':>13s}")
    print("-" * 64)
    for name in ARM_COLUMNS:
        if name == best:
            continue
        stats = paired_bootstrap(results[name], results[best])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:16s} {stats['improvement']:+12.5f} {ci:>20s} "
              f"{stats['p_better']:13.3f}")
    print(f"\nPositive favours {best}. An interval spanning zero means the two "
          f"arms are\nthe same feature, and the simpler one should win on cost.")


if __name__ == "__main__":
    main()
