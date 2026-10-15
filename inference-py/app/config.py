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

# DynamoDB holds current live-match state (PROJECT_SPEC.md Phase 3), written
# by ingestor-go's live poller (see ingestor-go/internal/live/dynamo_store.go)
# and read by app/db.py's get_live_state(). AWS_ENDPOINT_URL_DYNAMODB is
# picked up by boto3 itself when set (e.g. http://localhost:4566 for the
# LocalStack container in docker-compose.yml) - no extra plumbing needed here.
DYNAMODB_TABLE_NAME = os.environ["DYNAMODB_TABLE_NAME"]
AWS_REGION = os.environ["AWS_REGION"]

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

# Matches ml/train_v2.py: team form + club_goals_relative, saved locally to
# ml/models/match_outcome_model_v2.json and uploaded to the same S3 bucket
# under a different key so both artifacts coexist.
MODEL_V2_S3_KEY = os.environ.get("MODEL_V2_S3_KEY", "2000_v2_club_goals/model.json")
MODEL_V2_LOCAL_PATH = Path(
    os.environ.get(
        "MODEL_V2_LOCAL_PATH",
        Path(__file__).resolve().parents[2] / "ml" / "models" / "match_outcome_model_v2.json",
    )
)

# Matches ml/train_v3.py: v2's columns plus the seven `atkdef` rating columns,
# fitted at the retuned capacity ml/FINDINGS_CAPACITY.md settled on. Unlike
# train_v2.py, train_v3.py only uploads under --publish, so this key may not
# exist yet.
MODEL_V3_S3_KEY = os.environ.get("MODEL_V3_S3_KEY", "2000_v3_atkdef/model.json")
MODEL_V3_LOCAL_PATH = Path(
    os.environ.get(
        "MODEL_V3_LOCAL_PATH",
        Path(__file__).resolve().parents[2] / "ml" / "models" / "match_outcome_model_v3.json",
    )
)

# Matches ml/train_btts.py: v3's own 63 columns, retargeted at the BTTS label
# rather than 1X2. A fourth key, not a variant of v3's - different task, same
# recipe - so it coexists with whichever of v1/v2/v3 SERVED_MODEL_VERSION
# names rather than ever being a candidate for that slot.
MODEL_BTTS_S3_KEY = os.environ.get("MODEL_BTTS_S3_KEY", "2000_btts_v1/model.json")
MODEL_BTTS_LOCAL_PATH = Path(
    os.environ.get(
        "MODEL_BTTS_LOCAL_PATH",
        Path(__file__).resolve().parents[2] / "ml" / "models" / "btts_model_v1.json",
    )
)

# How long a replayed rating state may serve before app/ratings_state.py
# rebuilds it. The replay costs ~0.4s over the whole archive, so this trades a
# rare half-second hiccup on one request against how stale a rating may get;
# 0 disables refreshing entirely (useful in tests). An hour is well inside a
# fixture cadence - no league plays twice in one.
RATINGS_REFRESH_SECONDS = float(os.environ.get("RATINGS_REFRESH_SECONDS", "3600"))

# Short tag for the model that serves predictions and whose edges snapshots
# get graded, written to edges.model_version and carried into
# prediction_calls. Deliberately coarse ('v1'/'v2') rather than the exact
# artifact fingerprint model_v2.model_version() returns: reports slice on
# the tier, and an mtime-based fingerprint would start a new bucket on every
# retrain. backtest/stats.py --model reads this same value.
#
# v3 since the archive backfill made the comparison possible. The promotion
# rests on two offline lines, both measured on the chronological holdout
# neither artifact was fitted on (scripts/compare_v2_v3_served.py, 7,919
# fixtures):
#
#   v3 beats v2 by +0.00638 log loss on the inputs the service actually
#   produces, 95% CI [+0.00337, +0.00933], P(v3 better) 1.000. Feeding served
#   rather than trained inputs costs 5% of that margin, not the margin.
#
#   v3 carries 4.31% of its total gain on the two `club_goals_relative`
#   columns against v2's 23.43%. Those are the only inputs whose serving
#   definition cannot be made identical to their training one (see
#   db.get_league_goal_volume_reference), so the promoted model is also the
#   one less exposed to the project's one irreducible train/serve gap.
#
# Prerequisite, not incidental: v3's rating columns are replayed state, and
# that replay is only the trained one because `fixtures` now holds the whole
# archive. scripts/verify_archive_read.py is the gate that says so and exits
# non-zero when it stops being true.
SERVED_MODEL_VERSION = os.environ.get("SERVED_MODEL_VERSION", "v3")

# BTTS's own coarse tag, written to edges_btts.model_version - app/model_btts.py's
# model_version() returns an mtime-based fingerprint (so is_loaded() can tell
# two artifacts apart), but that is the wrong thing to key a backtest record
# on for the same reason SERVED_MODEL_VERSION is coarse rather than
# model_v3.model_version(): a retrain should not start a new bucket, and
# backtest/grade_calls.py's --model default has to be something an operator
# can type without looking up a timestamp first.
#
# Bumped to -blend when app/dc_state.py started feeding a classifier/Poisson
# blend rather than the bare classifier (ml/btts_ablate_v2.py: +0.00075 log
# loss over the classifier alone, CI excludes zero) - a different prediction
# deserves its own bucket, the same reason v2->v3 did.
SERVED_BTTS_MODEL_VERSION = os.environ.get("SERVED_BTTS_MODEL_VERSION", "btts-v1-blend")

# The Dixon-Coles arm app/dc_state.py replays to feed the Poisson half of the
# blend - build_goals_features.py's DIAGNOSTIC_ARM, the one `classifier` has
# actually been measured against, not ml/btts_ablate_v2.py's swept h730/c7
# (a marginal, not-significant improvement over this arm that costs 2.5x the
# replay time - not worth it for a null result).
DC_HALF_LIFE = float(os.environ.get("DC_HALF_LIFE", "365.0"))
DC_CADENCE = float(os.environ.get("DC_CADENCE", "30.0"))

# Weight on the classifier in the BTTS blend; (1 - this) on the Poisson arm.
# 0.7 per ml/btts_ablate_v2.py's rolling-origin sweep over {0.5, 0.6, 0.7,
# 0.8} - the point estimate peaks near 0.6-0.7 and 0.7 is the value measured
# against both the tuned and the untouched classifier/poisson pair.
BTTS_BLEND_WEIGHT = float(os.environ.get("BTTS_BLEND_WEIGHT", "0.7"))

# How long a replayed Dixon-Coles state may serve before app/dc_state.py
# rebuilds it in the background. The replay costs ~29s over the whole
# archive (vs ratings_state.py's 0.4s for atkdef - Dixon-Coles refits on a
# cadence, atkdef is one online pass) - too slow to pay inline on a request
# the way RATINGS_REFRESH_SECONDS does, so this drives a dedicated background
# thread instead. 0 disables refreshing (useful in tests).
DC_REFRESH_SECONDS = float(os.environ.get("DC_REFRESH_SECONDS", "3600"))

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

# Minimum |model_prob - market_prob| (on any one outcome) for /fixtures/upcoming
# to flag a fixture as an edge worth surfacing (PROJECT_SPEC.md Phase 4).
EDGE_FLAG_THRESHOLD = float(os.environ.get("EDGE_FLAG_THRESHOLD", "0.05"))
