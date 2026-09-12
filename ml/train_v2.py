"""Trains the v2 challenger: team form plus club_goals_relative (see
feature_lib.py's LeagueRollingMean/club_goals_relative), the single largest
squad-stats finding in ml/player_stats/FINDINGS_PHASE9.md that needs no
lineup data - only goals per club per match, which the incumbent already
ingests. Unlike ml/player_stats/train_challenger.py this is a real servable
model: it's saved to disk/S3 the same way ml/train.py's incumbent is, so
inference-py can load and serve it (see app/model_v2.py).

Originally mirrored ml/train.py's recipe and hyperparameters exactly, so that
any difference in validation performance was attributable to the two extra
feature columns and not to a differently-tuned model. That is no longer true:
the regularization constants below were tuned (ml/tune.py --grid reg/reg2) and
ml/train.py's incumbent has not had the same treatment applied, so a v1-vs-v2
gap read off the two *served* artifacts now measures features and fit together.

That sweep has been run against v1 (ml/tune.py --model v1 --grid reg,
2026-09-12) and the two effects turn out to be independent: v1 gains +0.00446
from the same regularization that gains v2 +0.00440, both picking the same
config, while the feature gap between them holds at +0.0054 (t = +3.3) whether
both are tuned or neither is. So club_goals_relative is worth ~0.0054 of log
loss on its own - but that figure only holds when both models are fitted the
same way, which the shipped artifacts currently are not.

Run:
    ./inference-py/bin/python ml/train_v2.py
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import boto3  # noqa: E402
import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

FEATURES_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"
MODEL_LOCAL_PATH = Path(__file__).parent / "models" / "match_outcome_model_v2.json"

# Same bucket as the incumbent (ml/train.py); a different key so both
# artifacts coexist.
S3_BUCKET_MODELS = os.environ.get("S3_BUCKET_MODELS", "liveedge-model-artifacts-590184129653")
S3_KEY = "2000_v2_club_goals/model.json"

MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"
EXPERIMENT_NAME = "liveedge-match-outcome-v2"

VAL_FRACTION = 0.2
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# Tree/rate constants, unchanged: a 36-point sweep over n_estimators, max_depth
# and learning_rate (ml/tune.py --grid core) found nothing that beat these
# beyond noise - 29 of 36 points separated from this recipe and every one of
# them was worse, so it already sits at a local optimum.
N_ESTIMATORS = 800
MAX_DEPTH = 4
LEARNING_RATE = 0.0072

# Regularization, tuned 2026-09-12 via ml/tune.py --grid reg (81 points) then
# --grid reg2 (32 points, extending the two parameters whose optimum had
# pinned to the first grid's edge). Worth +0.0044 val log loss against the
# unregularized recipe (0.99686 -> ~0.9925, paired t = +3.9), which is more
# than club_goals_relative itself bought over v1.
#
# REG_LAMBDA is the large lever and is a true interior optimum: the sweep saw
# 1 -> 10 -> 100 improve monotonically and 300 -> 1000 -> 3000 get worse.
# SUBSAMPLE is a real but smaller effect, and its curve is flat between 0.5 and
# 0.7; 0.6 is chosen over the nominally-best 0.5 because the two are
# statistically indistinguishable (0.0005 apart against an SE of ~0.0011) and
# 0.6 is not sitting on the edge of the grid that produced it.
#
# min_child_weight and colsample_bytree are deliberately absent: both were flat
# across their whole swept range (colsample spanned 0.0005 log loss, less than
# one SE), so they stay at XGBoost's defaults rather than being pinned to
# values that only won by chance.
REG_LAMBDA = 100
SUBSAMPLE = 0.6

# Pinned so the saved artifact is reproducible. XGBoost's default is also 0, so
# this changes nothing about the fit - it just stops a future default change
# from silently altering the model.
RANDOM_STATE = 0


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def main() -> None:
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date")

    form_cols = [c for c in df.columns if c.startswith("home_form_") or c.startswith("away_form_")]
    goals_cols = ["home_club_goals_relative", "away_club_goals_relative"]
    feature_cols = form_cols + goals_cols
    X = df[feature_cols]
    y = df["result"].map(LABEL_MAP).to_numpy()

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"train: {len(X_train)} rows, val: {len(X_val)} rows (chronological split)")
    print(f"{len(form_cols)} form features + {len(goals_cols)} goals-volume features")

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "reg_lambda": REG_LAMBDA,
            "subsample": SUBSAMPLE,
            "random_state": RANDOM_STATE,
            "val_fraction": VAL_FRACTION,
            "n_features": len(feature_cols),
            "n_train_rows": len(X_train),
            "n_val_rows": len(X_val),
        })

        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            reg_lambda=REG_LAMBDA,
            subsample=SUBSAMPLE,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)
        loss = log_loss(y_val, probs, labels=[0, 1, 2])
        brier = brier_score_multiclass(y_val, probs, n_classes=3)
        print(f"log loss: {loss:.5f}")
        print(f"brier score: {brier:.5f}")
        mlflow.log_metrics({"log_loss": loss, "brier_score": brier})

        mlflow.xgboost.log_model(model, name="v2_club_goals")

        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_LOCAL_PATH))
        print(f"saved model artifact to {MODEL_LOCAL_PATH}")

        # Publishes to whatever reads the bucket, so this is a deliberate step
        # rather than a side effect of training - comment it out again while
        # iterating on a model you aren't ready to ship. The bucket has
        # versioning enabled, so an upload that turns out to be wrong can be
        # rolled back to the previous version rather than being unrecoverable.
        boto3.client("s3").upload_file(str(MODEL_LOCAL_PATH), S3_BUCKET_MODELS, S3_KEY)
        mlflow.log_param("s3_key", S3_KEY)
        print(f"uploaded model artifact to s3://{S3_BUCKET_MODELS}/{S3_KEY}")

        print(f"run logged to mlflow experiment '{EXPERIMENT_NAME}' at {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
