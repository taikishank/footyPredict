"""Hyperparameter sweeps for the match-outcome models, logged to MLflow as one
parent run with a nested child run per grid point.

Covers both models (--model v1|v2). They share a feature table, a chronological
split and a baseline recipe, and differ only in which columns they read:

    v1  team form only                      -> experiment liveedge-match-outcome
    v2  team form + club_goals_relative      -> experiment liveedge-match-outcome-v2

Sweeping v1 on the same grid as v2 is what makes a v1-vs-v2 comparison
meaningful after v2 has been tuned. ml/train_v2.py deliberately mirrored
ml/train.py's hyperparameters so that any validation difference was
attributable to v2's two extra columns alone; the moment v2 is tuned and v1 is
not, that attribution breaks and "v2 is better" stops distinguishing a better
feature from a better fit. Running the identical grid against both restores it.

Two things this does that eyeballing `log_loss` across runs does not:

1. Paired significance. Two configs' mean log losses can differ in the 4th
   decimal and mean nothing - the val set's per-match log loss has a large
   spread, so the mean carries real sampling error. What matters is the
   *paired* per-match difference against a reference config, whose standard
   error is much smaller because the same matches are being compared. Each
   child run logs that difference, its SE, and a t-statistic.

2. No artifact clobbering. Nothing is written to either model's local path or
   to S3 - the served models are left alone. The single best config's model is
   logged to MLflow at the end, so promoting it later means loading that exact
   artifact rather than retraining and trusting the fit to reproduce.

Run:
    ./inference-py/bin/python ml/tune.py --model v2 --grid reg
    ./inference-py/bin/python ml/tune.py --model v1 --grid reg
    ./inference-py/bin/python ml/tune.py --model v1 --grid reg --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

FEATURES_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"
MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"

VAL_FRACTION = 0.2
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# Both models start from this recipe (ml/train.py and ml/train_v2.py carry the
# same three constants), so it is the paired reference for either sweep.
BASELINE = {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.0072}

# Which experiment each sweep lands in, matching the training script that owns
# the model, so a sweep sits alongside that model's own runs rather than in a
# third place. The extra v2 columns are named here rather than in the grids
# because they are a property of the model, not of the sweep.
MODELS = {
    "v1": {"experiment": "liveedge-match-outcome", "extra_cols": []},
    "v2": {
        "experiment": "liveedge-match-outcome-v2",
        "extra_cols": ["home_club_goals_relative", "away_club_goals_relative"],
    },
}

GRIDS = {
    "core": {
        "n_estimators": [400, 800, 1600],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.003, 0.0072, 0.02, 0.05],
    },
    "reg": {
        "min_child_weight": [1, 5, 20],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 10, 100],
    },
    # The "reg" sweep pinned its optimum to the grid's edge on reg_lambda
    # (monotone 1 -> 10 -> 100) and subsample (monotone 1.0 -> 0.7), so the
    # real optimum was outside it. This pushes both past where "reg" stopped,
    # and found reg_lambda=100 to be a true interior optimum after all.
    "reg2": {
        "subsample": [0.5, 0.6, 0.7, 0.85],
        "colsample_bytree": [0.6, 1.0],
        "reg_lambda": [100, 300, 1000, 3000],
    },
}

# Fixed so a rerun of the same config reproduces. The training scripts leave
# this at XGBoost's default, which is also 0, so the baseline row here matches
# what they produce.
RANDOM_STATE = 0

SIGNIFICANCE_T = 1.96  # two-sided 95%


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def per_match_nll(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Log loss per row, so two configs can be compared match-by-match."""
    return -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-15, None))


def fit_and_score(params: dict, X_train, y_train, X_val):
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        **params,
    )
    model.fit(X_train, y_train)
    return model, model.predict_proba(X_val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="v2",
                    help="which feature set to sweep (default: v2)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="core",
                    help="which named sweep to run (default: core)")
    ap.add_argument("--dry-run", action="store_true", help="list the grid, fit nothing")
    args = ap.parse_args()

    spec = MODELS[args.model]
    grid = GRIDS[args.grid]
    combos = [
        {**BASELINE, **dict(zip(grid.keys(), values))}
        for values in itertools.product(*grid.values())
    ]
    varied = list(grid.keys())

    if args.dry_run:
        for i, c in enumerate(combos, 1):
            print(f"{i:3d}. " + "  ".join(f"{k}={c[k]}" for k in varied))
        print(f"\n{len(combos)} grid point(s) in '{args.grid}' for model {args.model}; "
              f"held fixed at {BASELINE}")
        return

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date")
    form_cols = [c for c in df.columns if c.startswith(("home_form_", "away_form_"))]
    feature_cols = form_cols + spec["extra_cols"]
    X = df[feature_cols]
    y = df["result"].map(LABEL_MAP).to_numpy()

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"model {args.model} -> experiment '{spec['experiment']}'")
    print(f"train: {len(X_train)} rows, val: {len(X_val)} rows (chronological split)")
    print(f"val window: {df['date'].iloc[split_idx].date()} .. {df['date'].iloc[-1].date()}")
    print(f"{len(feature_cols)} features, {len(combos)} point(s) in grid '{args.grid}'")
    print(f"varying {varied}, holding {BASELINE}\n")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(spec["experiment"])

    with mlflow.start_run(run_name=f"sweep-{args.model}-{args.grid}-{len(combos)}pt") as parent:
        mlflow.log_params({
            "sweep": args.grid,
            "model": args.model,
            "n_grid_points": len(combos),
            "val_fraction": VAL_FRACTION,
            "n_features": len(feature_cols),
            "n_train_rows": len(X_train),
            "n_val_rows": len(X_val),
            "random_state": RANDOM_STATE,
            **{f"baseline_{k}": v for k, v in BASELINE.items()},
        })

        print("fitting baseline reference...", flush=True)
        _, base_probs = fit_and_score(BASELINE, X_train, y_train, X_val)
        base_nll = per_match_nll(base_probs, y_val)
        base_loss = float(base_nll.mean())
        mlflow.log_metric("baseline_log_loss", base_loss)
        print(f"baseline log loss: {base_loss:.5f}\n", flush=True)

        results = []
        best = None  # (log_loss, params, model)
        started = time.time()

        for i, params in enumerate(combos, 1):
            t0 = time.time()
            with mlflow.start_run(nested=True, run_name=f"{i:03d}"):
                model, probs = fit_and_score(params, X_train, y_train, X_val)
                nll = per_match_nll(probs, y_val)
                loss = log_loss(y_val, probs, labels=[0, 1, 2])
                brier = brier_score_multiclass(y_val, probs, n_classes=3)

                # Positive => this config beats the baseline on this val set.
                diff = base_nll - nll
                diff_mean = float(diff.mean())
                diff_se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                t_stat = diff_mean / diff_se if diff_se > 0 else 0.0

                mlflow.log_params({**params, "model": args.model,
                                   "random_state": RANDOM_STATE})
                mlflow.log_metrics({
                    "log_loss": loss,
                    "brier_score": brier,
                    "improvement_vs_baseline": diff_mean,
                    "improvement_se": diff_se,
                    "improvement_t": t_stat,
                })

                results.append({**params, "log_loss": loss, "brier": brier,
                                "diff": diff_mean, "se": diff_se, "t": t_stat})
                if best is None or loss < best[0]:
                    best = (loss, params, model)

            flag = "*" if abs(t_stat) > SIGNIFICANCE_T else " "
            desc = " ".join(f"{k}={params[k]:<5}" for k in varied)
            print(
                f"{i:3d}/{len(combos)} {desc} "
                f"ll={loss:.5f} vs base {diff_mean:+.5f} (t={t_stat:+.2f}){flag} "
                f"[{time.time() - t0:.1f}s]",
                flush=True,
            )

        print(f"\nswept {len(combos)} points in {time.time() - started:.0f}s")

        res = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
        print("\ntop 10 by val log loss ('*' = differs from baseline beyond noise):")
        for _, r in res.head(10).iterrows():
            flag = "*" if abs(r["t"]) > SIGNIFICANCE_T else " "
            desc = " ".join(f"{k}={r[k]:<5}" for k in varied)
            print(f"  {r['log_loss']:.5f} {flag} {desc} "
                  f"diff={r['diff']:+.5f} +/-{1.96 * r['se']:.5f}")

        best_loss, best_params, best_model = best
        mlflow.log_params({f"best_{k}": best_params[k] for k in varied})
        mlflow.log_metric("best_log_loss", best_loss)
        mlflow.xgboost.log_model(best_model, name=f"sweep_best_{args.model}_{args.grid}")

        n_sig = int((res["t"].abs() > SIGNIFICANCE_T).sum())
        best_desc = {k: best_params[k] for k in varied}
        print(f"\nbest: {best_desc}  log loss {best_loss:.5f} "
              f"({best_loss - base_loss:+.5f} vs baseline)")
        print(f"{n_sig}/{len(res)} grid point(s) differ from baseline beyond noise")
        print(f"\nbest model logged to mlflow experiment '{spec['experiment']}' "
              f"under parent run {parent.info.run_id}")
        print("nothing written to any model's local path or S3 - served models untouched")

        if n_sig == 0:
            print("\nNOTE: no config separated from the baseline. That is a result, not a "
                  "failure - it means this grid does not contain a real improvement.")
        print("CAUTION: the winner's log loss is selected on the same val split it is "
              "scored on, so it is optimistically biased. Treat it as a ranking, not as "
              "an unbiased estimate of forward performance.")


if __name__ == "__main__":
    main()
