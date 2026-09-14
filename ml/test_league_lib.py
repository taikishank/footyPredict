"""The two things that would silently turn this phase into a false positive:
a character column that has seen its own fixture's result, and a dummy set
that changes shape between folds."""
import math

import pandas as pd
import pytest

from league_lib import LeagueCharacter, build_character, league_dummies

LEAGUE = 8
OTHER = 82


def _fill(character, n, home_goals=2, away_goals=0, league=LEAGUE):
    for _ in range(n):
        character.update(league, home_goals, away_goals)


def test_nan_until_the_league_has_enough_history():
    character = LeagueCharacter(min_fixtures=200)
    _fill(character, 199)
    assert all(math.isnan(v) for v in character.features(LEAGUE).values())
    character.update(LEAGUE, 1, 0)
    assert not any(math.isnan(v) for v in character.features(LEAGUE).values())


def test_unseen_league_is_nan_not_zero():
    """Zero would read as 'a league where the home side never wins', which is
    a strong and false claim; NaN lets xgboost learn the missing branch."""
    character = LeagueCharacter()
    _fill(character, 300)
    assert all(math.isnan(v) for v in character.features(OTHER).values())


def test_rates_match_a_hand_count():
    character = LeagueCharacter(min_fixtures=4)
    for home, away in [(2, 0), (1, 1), (0, 3), (2, 2)]:
        character.update(LEAGUE, home, away)
    features = character.features(LEAGUE)
    assert features["league_home_rate"] == pytest.approx(0.25)
    assert features["league_draw_rate"] == pytest.approx(0.50)
    assert features["league_goals"] == pytest.approx(11 / 4)


def test_balance_separates_a_lopsided_league_from_a_tight_one():
    lopsided = LeagueCharacter(min_fixtures=2)
    tight = LeagueCharacter(min_fixtures=2)
    for home, away in [(5, 0), (0, 5), (4, 0), (0, 4)]:
        lopsided.update(LEAGUE, home, away)
    for home, away in [(1, 0), (0, 1), (1, 1), (2, 1)]:
        tight.update(LEAGUE, home, away)
    assert (lopsided.features(LEAGUE)["league_balance"]
            > tight.features(LEAGUE)["league_balance"])


def test_the_window_forgets():
    character = LeagueCharacter(window=10, min_fixtures=10)
    _fill(character, 10, home_goals=3, away_goals=0)
    assert character.features(LEAGUE)["league_home_rate"] == pytest.approx(1.0)
    _fill(character, 10, home_goals=0, away_goals=3)
    assert character.features(LEAGUE)["league_home_rate"] == pytest.approx(0.0)


def test_leagues_do_not_leak_into_each_other():
    character = LeagueCharacter(min_fixtures=2)
    _fill(character, 5, home_goals=3, away_goals=0, league=LEAGUE)
    _fill(character, 5, home_goals=0, away_goals=3, league=OTHER)
    assert character.features(LEAGUE)["league_home_rate"] == pytest.approx(1.0)
    assert character.features(OTHER)["league_home_rate"] == pytest.approx(0.0)


def test_a_fixture_never_sees_its_own_result():
    """The whole phase rests on this. Every fixture in the frame is a home win
    except the last; if build_character() updated before it described a
    fixture, the final row's home rate would be dragged by its own 0-1."""
    index = pd.DataFrame({
        "fixture_id": range(1, 302),
        "league_id": LEAGUE,
        "date": pd.date_range("2020-01-01", periods=301, freq="D"),
        "home_goals": [1] * 300 + [0],
        "away_goals": [0] * 300 + [1],
    })
    built = build_character(index).set_index("fixture_id")
    assert built.loc[301, "league_home_rate"] == pytest.approx(1.0)


def test_build_character_is_ordered_by_date_not_by_row():
    """results_index.parquet is not guaranteed sorted, and a pass that trusted
    row order would describe a fixture with fixtures played after it."""
    index = pd.DataFrame({
        "fixture_id": range(1, 302),
        "league_id": LEAGUE,
        "date": pd.date_range("2020-01-01", periods=301, freq="D"),
        "home_goals": [1] * 300 + [0],
        "away_goals": [0] * 300 + [1],
    })
    built = build_character(index.iloc[::-1]).set_index("fixture_id")
    assert built.loc[301, "league_home_rate"] == pytest.approx(1.0)


def test_dummies_cover_every_league_and_sum_to_one():
    ids = pd.Series([8, 82, 8, 564])
    dummies = league_dummies(ids)
    assert list(dummies.columns) == ["league_is_8", "league_is_82", "league_is_564"]
    assert (dummies.sum(axis=1) == 1).all()
    assert dummies["league_is_8"].tolist() == [1, 0, 1, 0]
