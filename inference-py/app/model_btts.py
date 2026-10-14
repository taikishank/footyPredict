"""Loads the BTTS challenger model (ml/train_btts.py's output) - v3's own 63
columns, retargeted at BTTS rather than 1X2, so it eats the same
features.v3_vector every fixture already builds for v3 serving. No second
feature vector, no second set of serving-time inputs to keep in sync with
training - see ml/train_btts.py's load() for why the column set is
identical.

Loading is best-effort, unlike app/model_v3.py's: BTTS is a second, optional
prediction surfaced alongside whichever 1X2 model SERVED_MODEL_VERSION
names, not a candidate to replace it, so its absence should not take the
app down. /fixtures/upcoming just omits the BTTS fields for every fixture
until ml/train_btts.py has been run and the artifact exists.
"""
from __future__ import annotations

import logging

import numpy as np
from xgboost import XGBClassifier

from app.config import MODEL_BTTS_LOCAL_PATH, MODEL_BTTS_S3_KEY, S3_BUCKET_MODELS

logger = logging.getLogger(__name__)

_model: XGBClassifier | None = None
_model_version: str | None = None


def load_model() -> None:
    """Loads the model into module state if it can - call once at app
    startup. Unlike model_v3.load_model(), a missing artifact logs and
    returns rather than raising."""
    global _model, _model_version

    path = MODEL_BTTS_LOCAL_PATH
    if not path.exists():
        if not S3_BUCKET_MODELS:
            logger.warning(
                "no BTTS model at %s and S3_BUCKET_MODELS is not set - "
                "/fixtures/upcoming will omit BTTS predictions until "
                "ml/train_btts.py has been run", path,
            )
            return
        import boto3

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            boto3.client("s3").download_file(S3_BUCKET_MODELS, MODEL_BTTS_S3_KEY, str(path))
        except Exception:
            logger.exception(
                "failed to download BTTS model from s3://%s/%s - "
                "/fixtures/upcoming will omit BTTS predictions",
                S3_BUCKET_MODELS, MODEL_BTTS_S3_KEY,
            )
            return

    model = XGBClassifier()
    model.load_model(str(path))
    _model = model
    _model_version = f"{path.name}@{path.stat().st_mtime_ns}"


def is_loaded() -> bool:
    return _model is not None


def model_version() -> str:
    if _model_version is None:
        raise RuntimeError("BTTS model not loaded")
    return _model_version


def predict_proba(vector: np.ndarray) -> dict[str, float]:
    if _model is None:
        raise RuntimeError("BTTS model not loaded - call load_model() first")
    yes = float(_model.predict_proba(vector)[0, 1])
    return {"yes": yes, "no": 1.0 - yes}
