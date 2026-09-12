"""Phase 9's offline comparison: does knowing the XI actually help?

Spec section 6 sets the bar - the challenger has to beat the incumbent "by more
than noise (e.g. a paired bootstrap over the validation set), not just 'adding
more features changed the number'". Two things are needed to make that bar
mean anything, and this script exists to enforce both:

1. **A rebaselined incumbent.** The shipped model trains on 39,594 fixtures
   back to 2000; a rating-aware challenger can only see the ~18,000 from 2016.
   Comparing the two directly would measure the lost decade, not the added
   features. So both arms are trained here, on identical rows and an identical
   chronological split, differing only in which columns they get.

2. **A paired test.** The two models are scored on the *same* validation
   fixtures, so the comparison resamples matched pairs of per-fixture losses
   rather than treating the two scores as independent samples. A tenth of a
   nat of log loss on 3,600 fixtures is easily within run-to-run noise;
   pairing is what makes a small real difference visible through it.

Run:
    ./inference-py/bin/python ml/player_stats/train_challenger.py --tier performance

Nothing here uploads to S3 or touches the incumbent's MLflow experiment.
Phase 9 is a decision, not a deployment; promotion is Phase 11-12's job.
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from player_feature_lib import AVAILABILITY_FIELD_ORDER  # noqa: E402

ML_DIR = Path(__file__).parent.parent
FEATURES_DIR = ML_DIR / "data" / "features"
MLFLOW_TRACKING_DIR = ML_DIR / "mlruns"

# Its own experiment, so the incumbent's run history stays directly comparable
# rather than being interleaved with runs on a different row count (spec 6).
EXPERIMENT_NAME = "liveedge-match-outcome-playerstats"

TIERS = {
    "availability": FEATURES_DIR / "challenger_availability.csv",
    "performance": FEATURES_DIR / "challenger_performance.csv",
}

VAL_FRACTION = 0.2
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}
BOOTSTRAP_SAMPLES = 2000

# Matched to ml/train.py, so the baseline arm here really is the incumbent
# recipe on fewer rows and not a differently-tuned model.
N_ESTIMATORS = 800
MAX_DEPTH = 4
LEARNING_RATE = 0.0072

RNG_SEED = 0


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    one_hot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def fit_and_score(X_train, y_train, X_val, y_val) -> tuple[np.ndarray, XGBClassifier]:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        eval_metric="mlogloss",
        random_state=RNG_SEED,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_val), model


def per_row_log_loss(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Each fixture's own contribution to log loss, which is what gets paired."""
    clipped = np.clip(probs, 1e-15, 1.0)
    return -np.log(clipped[np.arange(len(y_true)), y_true])


def paired_bootstrap(
    baseline_losses: np.ndarray, challenger_losses: np.ndarray
) -> dict[str, float]:
    """Resample matched per-fixture losses; positive `mean` favours the challenger.

    Resampling the *difference* rather than the two loss vectors separately is
    the point: both models saw the same fixtures, so a week of chaotic results
    inflates both losses together and should cancel, not add variance.
    """
    diff = baseline_losses - challenger_losses
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=TIERS, default="performance")
    args = parser.parse_args()

    df = pd.read_csv(TIERS[args.tier], parse_dates=["date"]).sort_values("date")

    form_cols = [c for c in df.columns if c.startswith(("home_form_", "away_form_"))]
    availability_cols = [
        f"{side}_sq_{name}"
        for side in ("home", "away")
        for name in AVAILABILITY_FIELD_ORDER
        if f"{side}_sq_{name}" in df.columns
    ]
    performance_cols = [
        c
        for c in df.columns
        if c.startswith(("home_sq_", "away_sq_")) and c not in availability_cols
    ]
    squad_cols = availability_cols + performance_cols
    y = df["result"].map(LABEL_MAP).to_numpy()

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"tier '{args.tier}': {len(df):,} rows "
          f"({df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d})")
    print(f"  train {len(y_train):,} / val {len(y_val):,}, {len(form_cols)} form "
          f"+ {len(availability_cols)} availability + {len(performance_cols)} performance")

    # Nested arms, each adding one block. The performance tier's real question
    # is not "do player stats beat team form" - the availability block already
    # answers that with a tenth of the columns - but "do ratings add anything
    # once we already know who is on the pitch". Only a nested comparison can
    # separate the two, and it is the one that decides whether the 2016 cutoff
    # and 36 extra columns are worth carrying.
    arms = {"baseline": form_cols, "availability": form_cols + availability_cols}
    if performance_cols:
        arms["performance"] = form_cols + squad_cols
    challenger = list(arms)[-1]
    results = {}
    for arm, cols in arms.items():
        X = df[cols]
        probs, model = fit_and_score(X.iloc[:split_idx], y_train, X.iloc[split_idx:], y_val)
        results[arm] = {
            "probs": probs,
            "model": model,
            "log_loss": log_loss(y_val, probs, labels=[0, 1, 2]),
            "brier": brier_score_multiclass(y_val, probs),
            "losses": per_row_log_loss(y_val, probs),
        }
        print(f"  {arm:11s} log loss {results[arm]['log_loss']:.5f}  "
              f"brier {results[arm]['brier']:.5f}")

    names = list(arms)
    comparisons = {
        f"{earlier}->{later}": paired_bootstrap(
            results[earlier]["losses"], results[later]["losses"]
        )
        for earlier, later in zip(names, names[1:])
    }
    if len(names) > 2:
        comparisons[f"baseline->{challenger}"] = paired_bootstrap(
            results["baseline"]["losses"], results[challenger]["losses"]
        )

    print(f"\npaired bootstrap ({BOOTSTRAP_SAMPLES} resamples), log loss reduction:")
    for label, test in comparisons.items():
        verdict = "clears" if test["ci_low"] > 0 else "within noise"
        print(f"  {label:28s} {test['improvement']:+.5f}  "
              f"95% CI [{test['ci_low']:+.5f}, {test['ci_high']:+.5f}]  "
              f"better in {test['p_better']:5.1%}  {verdict}")

    test = comparisons[f"baseline->{challenger}"]
    clears = test["ci_low"] > 0
    print(f"\n{challenger} tier {'CLEARS' if clears else 'DOES NOT CLEAR'} the section 6 "
          f"promotion bar against team form alone (95% CI excludes zero).")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{args.tier}-vs-baseline"):
        mlflow.log_params({
            "tier": args.tier,
            "arms": ",".join(names),
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "val_fraction": VAL_FRACTION,
            "n_form_features": len(form_cols),
            "n_availability_features": len(availability_cols),
            "n_performance_features": len(performance_cols),
            "n_train_rows": len(y_train),
            "n_val_rows": len(y_val),
        })
        mlflow.log_metrics({
            **{f"{arm}_log_loss": r["log_loss"] for arm, r in results.items()},
            **{f"{arm}_brier": r["brier"] for arm, r in results.items()},
            **{
                f"{label.replace('->', '_to_')}_{key}": value
                for label, test in comparisons.items()
                for key, value in test.items()
            },
            "clears_promotion_bar": float(clears),
        })
        importance = pd.Series(
            results[challenger]["model"].feature_importances_,
            index=arms[challenger],
        ).sort_values(ascending=False)
        print("\ntop 15 features in the challenger:")
        print(importance.head(15).to_string())
        mlflow.log_text(importance.to_string(), "feature_importance.txt")


if __name__ == "__main__":
    main()
