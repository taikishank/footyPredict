"""Reads the offline incumbent-vs-challenger comparison that
ml/player_stats/train_challenger.py logs to MLflow, for the frontend's model
comparison view.

The availability-tier challenger (team form + who's-on-the-pitch features) is
the one PLAYER_STATS_SPEC.md's Phase 9 findings say actually clears the
promotion bar - see ml/player_stats/FINDINGS_PHASE9.md. It isn't served live
yet (that needs the Phase 10 lineup poller), so this only ever reports its
offline validation performance against the incumbent, not live predictions.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
_MLFLOW_TRACKING_DIR = _ML_DIR / "mlruns"
_EXPERIMENT_NAME = "liveedge-match-outcome-playerstats"


class NoComparisonRunError(Exception):
    """No availability-tier train_challenger.py run has been logged yet."""


def latest_availability_comparison() -> dict:
    mlflow.set_tracking_uri(f"file:{_MLFLOW_TRACKING_DIR}")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
    if experiment is None:
        raise NoComparisonRunError(f"no MLflow experiment named {_EXPERIMENT_NAME!r}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="params.tier = 'availability'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise NoComparisonRunError("no availability-tier run logged yet")

    run = runs[0]
    metrics = run.data.metrics
    params = run.data.params

    # Two logging shapes exist across past runs of the script: an older one
    # (generic challenger_log_loss/ci_low/log_loss_improvement) and the
    # current one (baseline_to_availability_* + availability_log_loss), from
    # when multi-arm nested comparisons were added. Reading either lets this
    # work against whatever's actually the latest run without requiring a
    # rerun first.
    return {
        "run_id": run.info.run_id,
        "logged_at_ms": run.info.start_time,
        "n_train_rows": int(params["n_train_rows"]),
        "n_val_rows": int(params["n_val_rows"]),
        "incumbent_log_loss": metrics["baseline_log_loss"],
        "incumbent_brier": metrics["baseline_brier"],
        "challenger_log_loss": metrics.get("availability_log_loss", metrics.get("challenger_log_loss")),
        "challenger_brier": metrics.get("availability_brier", metrics.get("challenger_brier")),
        "log_loss_improvement": metrics.get(
            "baseline_to_availability_improvement", metrics.get("log_loss_improvement")
        ),
        "ci_low": metrics.get("baseline_to_availability_ci_low", metrics.get("ci_low")),
        "ci_high": metrics.get("baseline_to_availability_ci_high", metrics.get("ci_high")),
        "p_better": metrics.get("baseline_to_availability_p_better", metrics.get("p_better")),
        "clears_promotion_bar": bool(metrics["clears_promotion_bar"]),
    }
