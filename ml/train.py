import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import boto3
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

FEATURES_PATH = Path(__file__).parent / "data" / "features" / "all_leagues_features_removed_interceptions.csv"
CALIBRATION_PLOT_PATH = Path(__file__).parent / "models" / "calibration_curve.png"
MODEL_LOCAL_PATH = Path(__file__).parent / "models" / "match_outcome_model.json"

# Set by Terraform (infra/outputs.tf -> model_artifacts_bucket); override for local testing.
S3_BUCKET_MODELS = os.environ.get("S3_BUCKET_MODELS", "liveedge-model-artifacts-590184129653")

MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"
EXPERIMENT_NAME = "liveedge-match-outcome"

VAL_FRACTION = 0.2
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# Hyperparameters live here - edit these between runs, MLflow logs whatever
# values were actually used so nothing needs to go in a filename anymore.
N_ESTIMATORS = 150
MAX_DEPTH = 5
LEARNING_RATE = 0.0078


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def plot_calibration(y_true: np.ndarray, probs: np.ndarray) -> Path:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for label, class_idx in LABEL_MAP.items():
        frac_pos, mean_pred = calibration_curve(
            y_true == class_idx, probs[:, class_idx], n_bins=5, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend()
    CALIBRATION_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CALIBRATION_PLOT_PATH)
    plt.close(fig)
    return CALIBRATION_PLOT_PATH


def main() -> None:
    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"]).sort_values("date")

    feature_cols = [c for c in df.columns if c.startswith("home_form_") or c.startswith("away_form_")]
    X = df[feature_cols]
    y = df["result"].map(LABEL_MAP).to_numpy()

    split_idx = int(len(df) * (1 - VAL_FRACTION))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"train: {len(X_train)} rows, val: {len(X_val)} rows (chronological split)")

    with mlflow.start_run():
        mlflow.log_params({
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
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
            eval_metric="mlogloss",
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)
        loss = log_loss(y_val, probs, labels=[0, 1, 2])
        brier = brier_score_multiclass(y_val, probs, n_classes=3)
        print(f"log loss: {loss:.4f}")
        print(f"brier score: {brier:.4f}")
        mlflow.log_metrics({"log_loss": loss, "brier_score": brier})

        calibration_path = plot_calibration(y_val, probs)
        mlflow.log_artifact(str(calibration_path))

        mlflow.xgboost.log_model(model, name="removed_interceptions_model")

        run_id = mlflow.active_run().info.run_id
        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_LOCAL_PATH))

        # TODO(human): decide the S3 key naming/versioning scheme for this artifact
        # and assign it to `s3_key` below. You have `run_id`, `loss`, and `brier`
        # available here if you want any of them baked into the key.
        s3_key = "no_interceptions_v1/model.json"

        boto3.client("s3").upload_file(str(MODEL_LOCAL_PATH), S3_BUCKET_MODELS, s3_key)
        mlflow.log_param("s3_key", s3_key)
        print(f"uploaded model artifact to s3://{S3_BUCKET_MODELS}/{s3_key}")

        print(f"run logged to mlflow experiment '{EXPERIMENT_NAME}' at {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
