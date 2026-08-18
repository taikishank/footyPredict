"""Builds the model input row for a fixture from Postgres history, using the
same team_form logic ml/build_features.py used at training time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.db import FixtureRow, get_fixture, get_team_history
from feature_lib import FORM_FIELD_ORDER, MAX_WINDOW, MIN_WINDOW, team_form


class NotEnoughHistoryError(Exception):
    """Raised when a team has fewer than MIN_WINDOW prior fixtures in the DB."""

    def __init__(self, team_id: int, n_available: int):
        self.team_id = team_id
        self.n_available = n_available
        super().__init__(
            f"team {team_id} has only {n_available} prior fixtures (need >= {MIN_WINDOW})"
        )


@dataclass
class MatchFeatures:
    fixture: FixtureRow
    vector: np.ndarray  # shape (1, 2 * len(FORM_FIELD_ORDER)), column order matches training


def build_match_features(fixture_id: int) -> MatchFeatures | None:
    fixture = get_fixture(fixture_id)
    if fixture is None:
        return None

    home_history = get_team_history(fixture["home_id"], fixture["starting_at"], MAX_WINDOW)
    if len(home_history) < MIN_WINDOW:
        raise NotEnoughHistoryError(fixture["home_id"], len(home_history))

    away_history = get_team_history(fixture["away_id"], fixture["starting_at"], MAX_WINDOW)
    if len(away_history) < MIN_WINDOW:
        raise NotEnoughHistoryError(fixture["away_id"], len(away_history))

    home_form = team_form(home_history)
    away_form = team_form(away_history)

    row = [home_form[name] for name in FORM_FIELD_ORDER] + [
        away_form[name] for name in FORM_FIELD_ORDER
    ]
    return MatchFeatures(fixture=fixture, vector=np.array([row], dtype=float))
