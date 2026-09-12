from datetime import datetime

from pydantic import BaseModel


class Probabilities(BaseModel):
    home_win: float
    draw: float
    away_win: float


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
    # The v2 challenger's prediction (team form + club_goals_relative, see
    # app/model_v2.py) shown alongside the incumbent's - None under the same
    # condition as `probabilities`.
    probabilities_v2: Probabilities | None = None
    model_version_v2: str | None = None


class UpcomingFixturesResponse(BaseModel):
    generated_at: datetime
    window_days: int
    fixtures: list[UpcomingFixture]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None

