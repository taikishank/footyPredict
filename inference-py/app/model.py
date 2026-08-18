"""Loads the trained match-outcome model (ml/train.py's output) and wraps
prediction with the label names it was trained on."""
from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

from app.config import MODEL_LOCAL_PATH, MODEL_S3_KEY, S3_BUCKET_MODELS

# Must match ml/train.py's LABEL_MAP, ordered by class index (predict_proba's
# column order).
CLASS_LABELS = ["home_win", "draw", "away_win"]

_model: XGBClassifier | None = None
_model_version: str | None = None


def load_model() -> None:
    """Loads the model into module state - call once at app startup."""
    global _model, _model_version

    path = MODEL_LOCAL_PATH
    if not path.exists():
        if not S3_BUCKET_MODELS:
            raise RuntimeError(
                f"no model at {path} and S3_BUCKET_MODELS is not set - "
                "run ml/train.py or set S3_BUCKET_MODELS to download one"
            )
        import boto3

        path.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3").download_file(S3_BUCKET_MODELS, MODEL_S3_KEY, str(path))

    model = XGBClassifier()
    model.load_model(str(path))
    _model = model
    _model_version = f"{path.name}@{path.stat().st_mtime_ns}"


def is_loaded() -> bool:
    return _model is not None


def model_version() -> str:
    if _model_version is None:
        raise RuntimeError("model not loaded - call load_model() first")
    return _model_version


def predict_proba(vector: np.ndarray) -> dict[str, float]:
    if _model is None:
        raise RuntimeError("model not loaded - call load_model() first")
    probs = _model.predict_proba(vector)[0]
    return {label: float(p) for label, p in zip(CLASS_LABELS, probs)}
