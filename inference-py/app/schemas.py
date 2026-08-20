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
    # (NotEnoughHistoryError) - odds/edge fields aren't wired up until the
    # Odds API integration lands (PROJECT_SPEC.md Phase 4).
    probabilities: Probabilities | None
    model_version: str | None


class UpcomingFixturesResponse(BaseModel):
    generated_at: datetime
    window_days: int
    fixtures: list[UpcomingFixture]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
