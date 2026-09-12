from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Outcome = Literal["home_win", "draw", "away_win"]


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
    # The outcome the model calls: whichever of the three it rates most likely.
    prediction: Outcome
    model_version: str
    # True when at least one team has fewer than feature_lib.MIN_WINDOW prior
    # fixtures this season, so the form averages behind the prediction are
    # thinner than the model was trained on.
    low_confidence: bool = False


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
    # The model predicts every fixture it can build features for, regardless of
    # whether the market disagrees. Both are None only when no team has played
    # yet this season, or the feature build failed (NotEnoughHistoryError).
    probabilities: Probabilities | None
    prediction: Outcome | None = None
    model_version: str | None
    low_confidence: bool = False
    # market is None when the odds poller hasn't matched this fixture to a
    # bookmaker event yet. edge is additionally None when probabilities is
    # None, since it requires a model prediction to compare against
    # (PROJECT_SPEC.md Phase 4).
    market: Probabilities | None = None
    edge: Probabilities | None = None
    # A significant model-vs-market divergence worth betting into - a value
    # signal layered on top of the prediction, not a precondition for it.
    flagged: bool = False


class UpcomingFixturesResponse(BaseModel):
    generated_at: datetime
    window_days: int
    fixtures: list[UpcomingFixture]


class HealthResponse(BaseModel):
    status: str
    model_version: str | None = None
