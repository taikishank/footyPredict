"""v3's serving path: that the vector the service builds is the row the model
was trained on, and that a missing piece fails loudly instead of silently.

The column-order risk is the one worth testing. v3's vector is v2's columns
with seven rating columns appended, assembled in app/features.py from two
different sources; nothing about an XGBoost call would complain if they arrived
transposed, it would just return confident nonsense. So the check here is
against the real artifact's own idea of how wide its input is.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from app import features as features_mod
from app import model_v3, ratings_state, served
from app.features import MatchFeatures, build_match_features
from feature_lib import FORM_FIELD_ORDER, STAT_IDS

V2_WIDTH = 2 * len(FORM_FIELD_ORDER) + 2  # form both sides + club_goals_relative
V3_WIDTH = V2_WIDTH + len(ratings_state.FIELD_ORDER)

FIXTURE = {
    "fixture_id": 999,
    "league_id": 8,
    "starting_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "home_id": 1,
    "away_id": 2,
    "home_name": "Arsenal",
    "away_name": "Chelsea",
    "season_id": 2025,
}

_RATINGS = {name: float(i + 1) for i, name in enumerate(ratings_state.FIELD_ORDER)}


@pytest.fixture
def stub_sources(monkeypatch):
    """Everything build_match_features reads, stubbed to constants."""
    # Stats keyed off feature_lib's own list, so a new stat column added there
    # breaks this fixture loudly rather than leaving it silently short.
    stats = {name: 1.0 for name in STAT_IDS.values()}
    stats["shot_pct"] = 0.5
    stats["big_chances_conversion_rate"] = 0.5
    history = [{"goals_for": 1, "goals_against": 0, "stats": stats}] * 8
    monkeypatch.setattr(features_mod, "get_fixture", lambda fid: dict(FIXTURE))
    monkeypatch.setattr(features_mod, "get_team_history", lambda *a: history)
    monkeypatch.setattr(features_mod, "get_league_goal_volume_reference", lambda *a: 2.5)
    monkeypatch.setattr(features_mod, "get_team_goal_volume", lambda *a: 2.7)
    monkeypatch.setattr(ratings_state, "is_loaded", lambda: True)
    monkeypatch.setattr(ratings_state, "rating_features", lambda *a: dict(_RATINGS))


def test_v3_vector_appends_ratings_in_field_order(stub_sources):
    f = build_match_features(999)
    assert f.v3_vector.shape == (1, V3_WIDTH)
    # The first V2_WIDTH columns are v2's, untouched.
    np.testing.assert_array_equal(f.v3_vector[0, :V2_WIDTH], f.v2_vector[0])
    # The last seven are the rating columns, in FIELD_ORDER.
    expected = [_RATINGS[name] for name in ratings_state.FIELD_ORDER]
    np.testing.assert_array_equal(f.v3_vector[0, V2_WIDTH:], expected)


def test_v3_vector_is_none_without_rating_state(stub_sources, monkeypatch):
    monkeypatch.setattr(ratings_state, "is_loaded", lambda: False)
    f = build_match_features(999)
    assert f.v3_vector is None
    assert f.v2_vector is not None  # v2 must not be disturbed by v3's absence


def test_trained_artifact_accepts_the_served_vector(stub_sources):
    """The real check: load ml/train_v3.py's output and feed it a served row.

    Catches a width or ordering drift between training and serving that no
    amount of stubbing would - XGBoost validates the feature count against the
    booster it was fitted with.
    """
    from app.config import MODEL_V3_LOCAL_PATH

    if not MODEL_V3_LOCAL_PATH.exists():
        pytest.skip("no v3 artifact - run ml/train_v3.py")

    model_v3.load_model()
    f = build_match_features(999)
    probs = model_v3.predict_proba(f.v3_vector)

    assert set(probs) == {"home_win", "draw", "away_win"}
    assert sum(probs.values()) == pytest.approx(1.0)


def test_serving_without_a_vector_raises_a_readable_error(monkeypatch):
    """v3 configured but rating state missing must not reach numpy."""
    monkeypatch.setattr(served, "_vector_attr", "v3_vector")
    bare = MatchFeatures(fixture=dict(FIXTURE), vector=np.zeros((1, 2)), v3_vector=None)
    with pytest.raises(RuntimeError, match="no v3_vector for fixture 999"):
        served.predict_proba(bare)
