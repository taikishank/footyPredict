"""Builds the model input row for a fixture from Postgres history, using the
same team_form logic ml/build_features.py used at training time."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app import ratings_state
from app.db import (
    FixtureRow,
    get_fixture,
    get_league_goal_volume_reference,
    get_team_goal_volume,
    get_team_history,
)
from feature_lib import (
    FORM_FIELD_ORDER,
    GOALS_WINDOW,
    MAX_WINDOW,
    MIN_WINDOW,
    club_goals_relative,
    team_form,
)


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
    # v2 = vector's columns plus [home_club_goals_relative, away_club_goals_relative]
    # (ml/train_v2.py's column order) - NaN entries where a league reference
    # isn't available yet (XGBoost handles missing values natively). Optional
    # so callers that only need the incumbent's vector (tests, mainly) don't
    # have to fabricate one.
    v2_vector: np.ndarray | None = None
    # v3 = v2_vector's columns plus ratings_state.FIELD_ORDER (ml/train_v3.py's
    # column order, which is rating_ablate.load()'s base_cols + the atkdef arm).
    # None when the rating state has not been replayed - v3 is not the serving
    # model, so a caller that only needs v2 must not be made to pay for it.
    v3_vector: np.ndarray | None = None


def build_match_features(fixture_id: int) -> MatchFeatures | None:
    fixture = get_fixture(fixture_id)
    if fixture is None:
        return None

    # No season_id: get_team_history() spans season boundaries, matching
    # ml/build_features.py's window exactly (ml/FINDINGS_SCOPE.md).
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

    league_ref = get_league_goal_volume_reference(
        fixture["league_id"], fixture["starting_at"], GOALS_WINDOW
    )
    home_volume = get_team_goal_volume(fixture["home_id"], fixture["starting_at"], GOALS_WINDOW)
    away_volume = get_team_goal_volume(fixture["away_id"], fixture["starting_at"], GOALS_WINDOW)
    v2_row = row + [
        club_goals_relative(home_volume, league_ref),
        club_goals_relative(away_volume, league_ref),
    ]

    v3_vector = None
    if ratings_state.is_loaded():
        ratings = ratings_state.rating_features(
            fixture["league_id"], fixture["home_id"], fixture["away_id"],
            fixture["starting_at"],
        )
        v3_row = v2_row + [ratings[name] for name in ratings_state.FIELD_ORDER]
        v3_vector = np.array(
            [[v if v is not None else np.nan for v in v3_row]], dtype=float)

    return MatchFeatures(
        fixture=fixture,
        vector=np.array([row], dtype=float),
        v2_vector=np.array([[v if v is not None else np.nan for v in v2_row]], dtype=float),
        v3_vector=v3_vector,
    )
