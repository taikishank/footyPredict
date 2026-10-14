from datetime import datetime

from pydantic import BaseModel


class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float


class BTTSProbabilities(BaseModel):
    """Probabilities' two-outcome twin, for the BTTS market - a separate
    model class rather than reusing Probabilities with unused fields, since
    yes/no isn't a special case of home/draw/away."""

    yes: float
    no: float


class PredictionResponse(BaseModel):
    fixture_id: int
    league_id: int
    starting_at: datetime
    home_team: str
    away_team: str
    probabilities: Probabilities
    model_version: str


class LivePredictionResponse(PredictionResponse):
    state: str
    home_goals: int
    away_goals: int
    live_state_updated_at: datetime


class UpcomingFixture(BaseModel):
    fixture_id: int
    league_id: int
    starting_at: datetime
    home_team: str
    away_team: str
    # None when the team(s) don't have enough history yet for a prediction
    # (NotEnoughHistoryError).
    probabilities: Probabilities | None
    model_version: str | None
    # market is None when the odds poller hasn't matched this fixture to a
    # bookmaker event yet. edge is additionally None when probabilities is
    # None, since it requires a model prediction to compare against
    # (PROJECT_SPEC.md Phase 4).
    market: Probabilities | None = None
    edge: Probabilities | None = None
    flagged: bool = False
    # Legacy aliases of `probabilities`/`model_version`, kept only until the
    # frontend stops reading them. The name is now wrong in both halves: they
    # carry whatever SERVED_MODEL_VERSION names, not v2, and there is no
    # "challenger alongside the incumbent" here - see main.py's comment at the
    # assignment for why they follow the served model. None under the same
    # condition as `probabilities`.
    probabilities_v2: Probabilities | None = None
    model_version_v2: str | None = None
    # BTTS is a second, independent model/market on the same fixture (see
    # app/model_btts.py), not a variant of the 1X2 fields above - all four
    # are None together whenever the BTTS model isn't loaded or the fixture
    # has no v3-shaped feature vector to feed it, same condition that leaves
    # `probabilities` itself None.
    btts_probabilities: BTTSProbabilities | None = None
    btts_market: BTTSProbabilities | None = None
    btts_edge: BTTSProbabilities | None = None
    btts_flagged: bool = False


class UpcomingFixturesResponse(BaseModel):
    generated_at: datetime
    window_days: int
    fixtures: list[UpcomingFixture]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
    # Present once app/ratings_state.py has replayed. Carries n_results and
    # latest_result because those are what say whether the served ratings are
    # the trained ones - a count far from the training archive's means the
    # service is reading a different history than ml/ did.
    ratings: dict | None = None
    # False means /fixtures/upcoming is omitting BTTS fields, not that
    # anything is broken - see app/model_btts.py's docstring.
    btts_model_loaded: bool = False


class ModelPerformance(BaseModel):
    label: str
    log_loss: float
    brier: float


class ModelComparisonResponse(BaseModel):
    """Offline validation performance of the incumbent (team form) model
    against the availability-tier challenger, from the latest
    ml/player_stats/train_challenger.py run logged to MLflow. Not live
    predictions - the challenger isn't served yet (needs the Phase 10 lineup
    poller); see app/model_comparison.py."""

    run_id: str
    n_train_rows: int
    n_val_rows: int
    incumbent: ModelPerformance
    challenger: ModelPerformance
    log_loss_improvement: float
    ci_low: float
    ci_high: float
    p_better: float
    clears_promotion_bar: bool
