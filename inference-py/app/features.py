"""Builds the model input row for a fixture from Postgres history, using the
same team_form logic ml/build_features.py used at training time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.db import FixtureRow, get_fixture, get_team_history
from feature_lib import FORM_FIELD_ORDER, MAX_WINDOW, MIN_WINDOW, team_form


class NotEnoughHistoryError(Exception):
    """Raised when a team has no prior fixtures in the DB this season, leaving
    nothing to average into a form vector. A short history is not an error -
    see MatchFeatures.low_confidence."""

    def __init__(self, team_id: int, n_available: int):
        self.team_id = team_id
        self.n_available = n_available
        super().__init__(f"team {team_id} has no prior fixtures this season")


@dataclass
class MatchFeatures:
    fixture: FixtureRow
    vector: np.ndarray  # shape (1, 2 * len(FORM_FIELD_ORDER)), column order matches training
    # Fewest prior fixtures either team has this season. Below MIN_WINDOW the
    # form averages are noisier than anything the model saw during training,
    # so the prediction still stands but is marked low confidence.
    history_matches: int = MAX_WINDOW

    @property
    def low_confidence(self) -> bool:
        return self.history_matches < MIN_WINDOW


def build_match_features(fixture_id: int) -> MatchFeatures | None:
    fixture = get_fixture(fixture_id)
    if fixture is None:
        return None

    season_id = fixture["season_id"]
    home_history = get_team_history(
        fixture["home_id"], fixture["starting_at"], MAX_WINDOW, season_id
    )
    if not home_history:
        raise NotEnoughHistoryError(fixture["home_id"], 0)

    away_history = get_team_history(
        fixture["away_id"], fixture["starting_at"], MAX_WINDOW, season_id
    )
    if not away_history:
        raise NotEnoughHistoryError(fixture["away_id"], 0)

    home_form = team_form(home_history)
    away_form = team_form(away_history)

    row = [home_form[name] for name in FORM_FIELD_ORDER] + [
        away_form[name] for name in FORM_FIELD_ORDER
    ]
    return MatchFeatures(
        fixture=fixture,
        vector=np.array([row], dtype=float),
        history_matches=min(len(home_history), len(away_history)),
    )
