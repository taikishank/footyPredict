from datetime import datetime, timezone

import pytest

from app import features
from feature_lib import FORM_FIELD_ORDER, MIN_WINDOW, STAT_IDS

FIXTURE = {
    "fixture_id": 123,
    "league_id": 8,
    "starting_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "home_id": 1,
    "away_id": 2,
    "home_name": "Arsenal",
    "away_name": "Chelsea",
    "season_id": 25,
}


def _result(goals_for: int, goals_against: int) -> dict:
    stats = {name: 0.0 for name in STAT_IDS.values()}
    stats["shot_pct"] = 0.0
    stats["big_chances_conversion_rate"] = 0.0
    return {"goals_for": goals_for, "goals_against": goals_against, "stats": stats}


@pytest.fixture
def stub_db(monkeypatch):
    """Stubs the two db lookups build_match_features makes; `counts` sets how
    many prior results each team has."""

    def _stub(counts: dict[int, int], fixture=FIXTURE):
        monkeypatch.setattr(features, "get_fixture", lambda fixture_id: fixture)
        monkeypatch.setattr(
            features,
            "get_team_history",
            lambda team_id, before, limit, season_id: [_result(1, 0)] * counts[team_id],
        )

    return _stub


def test_predicts_with_a_full_window(stub_db):
    stub_db({1: 8, 2: 8})

    built = features.build_match_features(123)

    assert built is not None
    assert built.vector.shape == (1, 2 * len(FORM_FIELD_ORDER))
    assert built.history_matches == 8
    assert built.low_confidence is False


def test_predicts_with_a_partial_window_but_flags_low_confidence(stub_db):
    """Short history is a confidence caveat, not a reason to withhold the
    prediction."""
    stub_db({1: 1, 2: MIN_WINDOW})

    built = features.build_match_features(123)

    assert built is not None
    assert built.vector.shape == (1, 2 * len(FORM_FIELD_ORDER))
    assert built.history_matches == 1
    assert built.low_confidence is True


def test_raises_when_a_team_has_no_history_at_all(stub_db):
    stub_db({1: 4, 2: 0})

    with pytest.raises(features.NotEnoughHistoryError) as excinfo:
        features.build_match_features(123)

    assert excinfo.value.team_id == 2


def test_returns_none_when_fixture_missing(monkeypatch):
    monkeypatch.setattr(features, "get_fixture", lambda fixture_id: None)

    assert features.build_match_features(999) is None
