"""Runtime configuration for the inference service, loaded from the
environment (.env locally, real env vars on ECS)."""
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

# ml/feature_lib.py is the source of truth for feature engineering, shared
# with the training pipeline - see ml/feature_lib.py's module docstring.
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

POSTGRES_URL = os.environ["POSTGRES_URL"]

# Matches ml/train.py: it saves locally to ml/models/match_outcome_model.json
# and uploads the same file to s3://{S3_BUCKET_MODELS}/{MODEL_S3_KEY}.
S3_BUCKET_MODELS = os.environ.get("S3_BUCKET_MODELS", "")
MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY", "2000_v1/model.json")
MODEL_LOCAL_PATH = Path(
    os.environ.get(
        "MODEL_LOCAL_PATH",
        Path(__file__).resolve().parents[2] / "ml" / "models" / "match_outcome_model.json",
    )
)

# Angular dev server origins allowed to call this API cross-origin (see
# app.ts's `ng serve` / environment.ts's apiBaseUrl). Comma-separated;
# override via env once there's a real deployed frontend origin.
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ALLOWED_ORIGINS", "http://localhost:4200,http://localhost:4300"
    ).split(",")
    if origin.strip()
]
