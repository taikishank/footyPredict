"""The contract OpponentsFaced has to hold to be readable next to the form
columns: the same matches, in the same order, with no sight of the fixture
being predicted."""
from collections import deque

import pytest

from feature_lib import MAX_WINDOW, STAT_IDS, team_form
from form_lib import FACED_FIELD_ORDER, OpponentsFaced


def _zero_stats() -> dict:
    names = list(STAT_IDS.values()) + ["shot_pct", "big_chances_conversion_rate"]
    return {name: 0.0 for name in names}


def test_no_history_is_none():
    assert OpponentsFaced().faced(1) is None


def test_means_the_recorded_opponents():
    faced = OpponentsFaced()
    faced.record(1, 0.2, 0.4)
    faced.record(1, 0.4, 0.8)
    assert faced.faced(1) == {"opp_atk_faced": pytest.approx(0.3),
                              "opp_def_faced": pytest.approx(0.6)}


def test_field_order_matches_the_keys_emitted():
    faced = OpponentsFaced()
    faced.record(1, 0.0, 0.0)
    assert tuple(faced.faced(1)) == FACED_FIELD_ORDER


def test_clubs_are_independent():
    faced = OpponentsFaced()
    faced.record(1, 1.0, 1.0)
    faced.record(2, -1.0, -1.0)
    assert faced.faced(1)["opp_atk_faced"] == 1.0
    assert faced.faced(2)["opp_atk_faced"] == -1.0


def test_window_drops_the_oldest_like_team_form():
    """The binding property: over the same match sequence, the opponents-faced
    mean must cover exactly the matches team_form() averages. If the two
    windows disagree the columns describe different schedules."""
    faced = OpponentsFaced()
    history = []
    for i in range(MAX_WINDOW + 4):
        faced.record(1, float(i), 0.0)
        history.append({"goals_for": float(i), "goals_against": 0.0,
                        "stats": _zero_stats()})

    form = team_form(history)
    assert faced.faced(1)["opp_atk_faced"] == pytest.approx(form["goals_for"])


def test_honours_a_non_default_window():
    faced = OpponentsFaced(window_size=2)
    for value in (1.0, 2.0, 3.0):
        faced.record(1, value, 0.0)
    assert faced.faced(1)["opp_atk_faced"] == pytest.approx(2.5)
    assert isinstance(faced._history[1], deque)
