import pytest

from app.live_prob import adjust_for_live_state

PRIOR = {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}


def test_probabilities_still_sum_to_one():
    probs = adjust_for_live_state(PRIOR, state="2H", home_goals=2, away_goals=0)
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0


def test_level_score_leaves_probabilities_unchanged():
    probs = adjust_for_live_state(PRIOR, state="2H", home_goals=1, away_goals=1)
    for label in PRIOR:
        assert probs[label] == pytest.approx(PRIOR[label], abs=1e-9)


def test_home_lead_raises_home_win_and_lowers_draw_and_away():
    probs = adjust_for_live_state(PRIOR, state="2H", home_goals=1, away_goals=0)
    assert probs["home_win"] > PRIOR["home_win"]
    assert probs["draw"] < PRIOR["draw"]
    assert probs["away_win"] < PRIOR["away_win"]


def test_same_lead_matters_more_late_than_early():
    early = adjust_for_live_state(PRIOR, state="1H", home_goals=1, away_goals=0)
    late = adjust_for_live_state(PRIOR, state="2H", home_goals=1, away_goals=0)
    assert late["home_win"] > early["home_win"]


def test_unknown_state_falls_back_to_default_without_error():
    probs = adjust_for_live_state(PRIOR, state="BREAK", home_goals=0, away_goals=0)
    assert pytest.approx(sum(probs.values()), abs=1e-9) == 1.0
