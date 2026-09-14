"""The scoping recipes, against the two implementations they are replaying.

The point of these is that `window(cross_season=False)` really is what
`inference-py/app/db.py:76` does and `window(cross_season=True)` really is
what `ml/build_features.py:96` does. If either drifts, the ablation measures
a recipe nothing runs.
"""
from __future__ import annotations

import pytest

import scope_lib
from feature_lib import MAX_WINDOW, MIN_WINDOW


def hist(*season_ids: int) -> list[dict]:
    """Most-recent-last history, one entry per season id given."""
    return [{"season_id": s, "goals_for": 0, "goals_against": 0, "stats": {}} for s in season_ids]


# --- window ---------------------------------------------------------------

def test_cross_season_is_the_training_slice():
    """build_features.py's line is history[-MAX_WINDOW:], nothing else."""
    h = hist(*([1] * 5 + [2] * 5))
    assert scope_lib.window(h, 2, cross_season=True) == h[-MAX_WINDOW:]


def test_scoped_drops_prior_seasons_before_limiting():
    """db.py filters on season_id and *then* applies LIMIT, so a club ten
    matches into a season gets its last MAX_WINDOW in-season matches - not
    the in-season subset of its last MAX_WINDOW matches overall."""
    h = hist(*([1] * 20 + [2] * 10))
    w = scope_lib.window(h, 2, cross_season=False)
    assert len(w) == MAX_WINDOW
    assert all(e["season_id"] == 2 for e in w)


def test_scoped_returns_a_partial_window_early_in_a_season():
    """The distribution half of the skew: the service averages three matches
    where the training table averages eight."""
    h = hist(*([1] * 30 + [2] * 3))
    assert len(scope_lib.window(h, 2, cross_season=False)) == 3
    assert len(scope_lib.window(h, 2, cross_season=True)) == MAX_WINDOW


def test_scoped_is_empty_at_a_season_opener():
    assert scope_lib.window(hist(1, 1, 1), 2, cross_season=False) == []


def test_the_two_recipes_agree_once_the_season_is_long_enough():
    """From the ninth in-season match on there is no skew left to measure -
    which is why the effect has to be looked for by month, not in the mean."""
    h = hist(*([1] * 10 + [2] * MAX_WINDOW))
    assert (scope_lib.window(h, 2, cross_season=True)
            == scope_lib.window(h, 2, cross_season=False))


# --- staleness ------------------------------------------------------------

def test_staleness_counts_the_season_not_the_window():
    """matches_into_season keeps climbing after the window is full; that is
    the "how far into the season" signal, and it is deliberately not capped."""
    h = hist(*([1] * 20 + [2] * 12))
    assert scope_lib.staleness(h, 2) == {"matches_into_season": 12, "stale_slots": 0}


def test_stale_slots_counts_carried_over_matches_in_the_window():
    h = hist(*([1] * 20 + [2] * 3))
    assert scope_lib.staleness(h, 2) == {
        "matches_into_season": 3,
        "stale_slots": MAX_WINDOW - 3,
    }


def test_staleness_at_a_season_opener_is_a_full_stale_window():
    h = hist(*([1] * 20))
    assert scope_lib.staleness(h, 2) == {"matches_into_season": 0, "stale_slots": MAX_WINDOW}


def test_staleness_fields_match_the_declared_column_names():
    """build_scope_features.py emits these as f"{side}_{name}" and
    scope_ablate.py reads them back by the same constant."""
    assert set(scope_lib.staleness(hist(1), 1)) == set(scope_lib.STALENESS_FIELDS)


# --- eligibility ----------------------------------------------------------

@pytest.mark.parametrize("n_in_season,expected", [
    (0, False), (MIN_WINDOW - 1, False), (MIN_WINDOW, True), (MAX_WINDOW, True),
])
def test_scoped_gate_counts_in_season_matches_only(n_in_season, expected):
    """NotEnoughHistoryError is raised on the length of the season-scoped
    query's result, however much history the club has overall."""
    h = hist(*([1] * 30 + [2] * n_in_season))
    assert scope_lib.eligible(h, 2, scope_lib.RECIPES["scoped"]) is expected


def test_xseason_gate_is_satisfied_by_last_season_alone():
    """The coverage half of the skew, in one assertion: a club with a full
    prior season and no matches yet in this one is a training row and is not
    a servable fixture."""
    h = hist(*([1] * 30))
    assert scope_lib.eligible(h, 2, scope_lib.RECIPES["xseason"])
    assert not scope_lib.eligible(h, 2, scope_lib.RECIPES["scoped"])


def test_full8_refuses_what_scoped_would_average():
    h = hist(*([1] * 30 + [2] * (MAX_WINDOW - 1)))
    assert scope_lib.eligible(h, 2, scope_lib.RECIPES["scoped"])
    assert not scope_lib.eligible(h, 2, scope_lib.RECIPES["scoped_full8"])


def test_fixture_eligibility_gates_on_the_weaker_club():
    """Both clubs must clear the gate - one debutant is enough to drop the
    fixture, on either path."""
    veteran, newcomer = hist(*([2] * MAX_WINDOW)), hist(*([1] * 30))
    flags = scope_lib.fixture_eligibility(veteran, newcomer, 2)
    assert flags["xseason"] and not flags["scoped"] and not flags["scoped_full8"]


def test_the_age_arm_is_eligible_exactly_where_the_training_recipe_is():
    """xseason_age changes the columns, not the row set - otherwise its
    comparison against xseason would not be a paired one."""
    for h in (hist(*([1] * 30)), hist(*([1] * 30 + [2] * 4)), hist(1, 1, 1)):
        flags = scope_lib.fixture_eligibility(h, h, 2)
        assert flags["xseason"] == flags["xseason_age"]


def test_every_recipe_reads_an_emitted_form_column_set():
    """build_scope_features.py emits exactly FORM_SOURCES; a recipe naming
    anything else would select columns that are not in the table."""
    assert {r.form_source for r in scope_lib.RECIPES.values()} == {"xseason", "scoped"}
