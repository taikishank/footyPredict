"""Trains a both-teams-to-score classifier on v3's own feature set.

A new target on an old table, deliberately: the 63 columns (`ra.load()`'s
form/scope/venue/opponent base plus `atkdef`'s seven rating columns) were
picked for the 1X2 task, and whether they carry anything for BTTS is the
open question `ml/btts_ablate.py` answers - this script does not re-open
the column choice, it just fits the recipe v3 already settled on against a
different label.

The label is `home_goals > 0 and away_goals > 0`, read off
`results_index.parquet` and joined on `fixture_id` - the same raw columns
`goals_lib.py`'s Dixon-Coles fit uses, so the two BTTS arms this phase
compares are trained on identical rows.

No retuning: `train_v3.PARAMS` (depth 2, 800 rounds, reg_lambda 30) is
reused as-is rather than swept for the new target, since nothing has yet
shown the 1X2 capacity retune doesn't transfer. `ml/btts_ablate.py`'s
classifier arm is this same recipe; this script only adds artifact
serialisation and the --publish path on top.

Run:
    ./inference-py/bin/python ml/train_btts.py
    ./inference-py/bin/python ml/train_btts.py --publish
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import rating_ablate as ra  # noqa: E402
import train_v3 as tv3  # noqa: E402
from build_goals_features import INDEX_PATH  # noqa: E402

MODEL_LOCAL_PATH = Path(__file__).parent / "models" / "btts_model_v1.json"

# A fourth key alongside v1/v2/v3's, since this is a different task (binary
# BTTS, not 1X2) rather than a challenger to any of them.
S3_BUCKET_MODELS = os.environ.get("S3_BUCKET_MODELS", "liveedge-model-artifacts-590184129653")
S3_KEY = "2000_btts_v1/model.json"

MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"
EXPERIMENT_NAME = "liveedge-btts-v1"

# v3's recipe, unmodified - see the module docstring for why this is not a
# retune.
PARAMS = tv3.PARAMS


def build() -> XGBClassifier:
    """An unfitted BTTS classifier. One constructor, so the artifact and
    ml/btts_ablate.py's classifier arm cannot drift apart."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        **PARAMS,
    )


def load() -> tuple:
    """v3's feature table, with a BTTS label joined on from the raw results
    index rather than the `result` column v3's own target comes from -
    `result` is categorical (home_win/draw/away_win) and does not carry the
    goal counts BTTS needs.
    """
    df, base_cols = ra.load()
    cols = base_cols + ra.ARM_COLUMNS[tv3.RATING_ARM]
    goals = pd.read_parquet(INDEX_PATH)[["fixture_id", "home_goals", "away_goals"]]
    before = len(df)
    df = df.merge(goals, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, (
        f"{before - len(df)} fixtures in v3's table have no row in "
        f"{INDEX_PATH.name}"
    )
    y = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).to_numpy(dtype=int)
    return df, cols, y


def score(df, cols, y, fold_list) -> tuple[float, float, XGBClassifier]:
    """Pooled log loss and Brier over every fold's validation slice, binary's
    twin of train_v3.score()."""
    parts, model = [], None
    for train_idx, val_idx in fold_list:
        model = build()
        model.fit(df[cols].iloc[train_idx], y[train_idx])
        parts.append(model.predict_proba(df[cols].iloc[val_idx])[:, 1])
    probs = np.concatenate(parts)
    val_all = np.concatenate([val for _, val in fold_list])
    y_val = y[val_all]
    brier = float(np.mean((probs - y_val) ** 2))
    return log_loss(y_val, probs, labels=[0, 1]), brier, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true",
                    help="upload the artifact to S3. Off by default: training a "
                         "model and shipping it are separate decisions.")
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, cols, y = load()
    split_folds = ra.folds(len(df), "split")
    rolling_folds = ra.folds(len(df), "rolling")

    (train_idx, val_idx), = split_folds
    print(f"{len(cols)} features (v3's own set, unmodified)")
    print(f"train: {len(train_idx)} rows, val: {len(val_idx)} rows "
          f"(chronological split)")
    print(f"BTTS rate: train {y[train_idx].mean():.4f}, val {y[val_idx].mean():.4f}")
    print(f"recipe: {PARAMS}\n")

    with mlflow.start_run():
        mlflow.log_params({
            **PARAMS,
            "rating_arm": tv3.RATING_ARM,
            "n_features": len(cols),
            "n_rows": len(df),
            "n_train_rows": len(train_idx),
            "n_val_rows": len(val_idx),
        })

        loss, brier, model = score(df, cols, y, split_folds)
        print(f"split    log loss {loss:.5f}   brier {brier:.5f}")
        mlflow.log_metrics({"log_loss": loss, "brier_score": brier})

        r_loss, r_brier, _ = score(df, cols, y, rolling_folds)
        print(f"rolling  log loss {r_loss:.5f}   brier {r_brier:.5f}  "
              f"({len(rolling_folds)} folds)")
        mlflow.log_metrics({"rolling_log_loss": r_loss,
                            "rolling_brier_score": r_brier})

        mlflow.xgboost.log_model(model, name="btts_v1")

        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_LOCAL_PATH))
        print(f"\nsaved model artifact to {MODEL_LOCAL_PATH}")

        if args.publish:
            import boto3
            boto3.client("s3").upload_file(
                str(MODEL_LOCAL_PATH), S3_BUCKET_MODELS, S3_KEY)
            mlflow.log_param("s3_key", S3_KEY)
            print(f"uploaded model artifact to s3://{S3_BUCKET_MODELS}/{S3_KEY}")
        else:
            print("not published (pass --publish to upload to "
                  f"s3://{S3_BUCKET_MODELS}/{S3_KEY})")

        print(f"run logged to mlflow experiment '{EXPERIMENT_NAME}' at {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
