import math

import pandas as pd
import pytest

from ratings_lib import AttackDefence
from venue_lib import (
    HFA_MIN_FIXTURES,
    AdaptiveHFA,
    PlaceboVenue,
    RollingHFA,
    VenueAttackDefence,
    VENUE_ARMS,
)

LEAGUE = 8


def _found_league(rating, team_ids=(1, 2, 3, 4)):
    """A round robin, so the league has a settled cohort before anyone is
    promoted into it - test_ratings_lib.py's helper, for the same reason."""
    for home in team_ids:
        for away in team_ids:
            if home != away:
                rating.update(home, away, LEAGUE, 1, 1)
    return rating


# --- the contracts every arm has to hold ------------------------------------

@pytest.mark.parametrize("name", VENUE_ARMS)
def test_field_order_matches_emitted_keys(name):
    rating = VENUE_ARMS[name]()
    assert tuple(rating.features(1, 2, LEAGUE, 500)) == rating.FIELD_ORDER


@pytest.mark.parametrize("name", VENUE_ARMS)
def test_expected_goals_stay_finite_under_a_runaway_streak(name):
    rating = _found_league(VENUE_ARMS[name]())
    for i in range(500):
        rating.update(1, 2, LEAGUE, 9, 0, 1000 + i)
    feats = rating.features(1, 2, LEAGUE, 9999)
    assert feats["exp_goals_home"] <= AttackDefence.MAX_LAMBDA
    assert abs(feats["home_atk"]) <= AttackDefence.MAX_RATING


@pytest.mark.parametrize("name", VENUE_ARMS)
def test_recenter_preserves_expected_goal_difference(name):
    """recenter() keeps the level columns off the date; it must not move the
    rates the model actually uses."""
    rating = _found_league(VENUE_ARMS[name]())
    rating.update(1, 2, LEAGUE, 4, 0, 501)
    rating.update(3, 4, LEAGUE, 0, 2, 502)
    before = rating.features(1, 2, LEAGUE, 600)
    rating.recenter(LEAGUE)
    after = rating.features(1, 2, LEAGUE, 600)
    assert after["exp_goal_diff"] == pytest.approx(before["exp_goal_diff"], abs=1e-9)


# --- the served-rate control ------------------------------------------------

def test_adaptive_hfa_at_the_served_rate_is_the_served_rating():
    """The assert the whole phase's readability rests on, in miniature: the
    lr arms must differ from v3 in the home-advantage rate and nothing else.
    build_venue_features.py makes the same check over all 40,098 fixtures."""
    served, adaptive = AttackDefence(), AdaptiveHFA(hfa_rate=0.01)
    for i, (h, a, hg, ag) in enumerate([(1, 2, 2, 1), (3, 4, 0, 0), (2, 3, 1, 3),
                                        (4, 1, 2, 2), (1, 3, 5, 0)] * 6):
        assert served.features(h, a, LEAGUE) == pytest.approx(
            adaptive.features(h, a, LEAGUE, i))
        served.update(h, a, LEAGUE, hg, ag)
        adaptive.update(h, a, LEAGUE, hg, ag, i)


def test_a_higher_rate_moves_home_advantage_further_on_the_same_evidence():
    slow, fast = AdaptiveHFA(hfa_rate=0.01), AdaptiveHFA(hfa_rate=1.0)
    for i in range(60):
        # Away sides winning everywhere: home advantage should fall, and fall
        # faster for the arm that is allowed to move.
        for rating in (slow, fast):
            rating.update(1, 2, LEAGUE, 0, 3, i)
    assert fast.hfa[LEAGUE] < slow.hfa[LEAGUE] < AttackDefence.INIT_HFA


# --- the rolling estimator --------------------------------------------------

def test_rolling_hfa_falls_back_to_the_seed_until_it_has_enough_history():
    rolling = RollingHFA()
    assert rolling._rolling_hfa(LEAGUE) == AttackDefence.INIT_HFA
    for i in range(HFA_MIN_FIXTURES - 1):
        rolling.update(1, 2, LEAGUE, 3, 1, i)
    assert rolling._rolling_hfa(LEAGUE) == AttackDefence.INIT_HFA
    rolling.update(1, 2, LEAGUE, 3, 1, HFA_MIN_FIXTURES)
    assert rolling._rolling_hfa(LEAGUE) != AttackDefence.INIT_HFA


def test_rolling_hfa_recovers_the_leagues_own_log_ratio():
    rolling = RollingHFA()
    for i in range(HFA_MIN_FIXTURES + 20):
        rolling.update(1, 2, LEAGUE, 2, 1, i)
    assert rolling._rolling_hfa(LEAGUE) == pytest.approx(math.log(2.0) - math.log(1.0))


def test_rolling_hfa_forgets_an_era_once_it_leaves_the_window():
    """The property a learning rate only approximates, and the reason the arm
    exists: 2020's empty stadiums have to leave the estimate on a known
    timescale rather than an asymptotic one."""
    rolling = RollingHFA(window=100, min_fixtures=50)
    for i in range(100):
        rolling.update(1, 2, LEAGUE, 3, 1, i)
    high = rolling._rolling_hfa(LEAGUE)
    for i in range(100):
        rolling.update(1, 2, LEAGUE, 1, 1, 1000 + i)
    assert rolling._rolling_hfa(LEAGUE) == pytest.approx(0.0)
    assert high > 0.5


# --- the venue split and its control ----------------------------------------

def test_a_home_result_does_not_move_the_clubs_away_ratings():
    """What the venue split *is*. If this fails the arm is not measuring
    venue at all."""
    venue = _found_league(VenueAttackDefence())
    before = venue.atk[(1, VenueAttackDefence.AWAY)]
    venue.update(1, 2, LEAGUE, 5, 0, 700)
    assert venue.atk[(1, VenueAttackDefence.HOME)] > before
    assert venue.atk[(1, VenueAttackDefence.AWAY)] == before


def test_the_placebo_has_the_same_shape_but_splits_on_the_fixture_id():
    """Same parameter count, same halving of the evidence, no football in the
    split - so a gain the placebo also gets is about column geometry rather
    than about playing at home. FINDINGS_LEAGUE.md's `interact` is why."""
    placebo = PlaceboVenue()
    assert placebo._slots(2) == (placebo.HOME, placebo.AWAY)
    assert placebo._slots(3) == (placebo.AWAY, placebo.HOME)

    venue = VenueAttackDefence()
    _found_league(placebo)
    _found_league(venue)
    assert len(placebo.atk) == len(venue.atk)

    # An odd-numbered fixture sends the home side's result to the slot the
    # venue arm would have called "away".
    placebo.update(1, 2, LEAGUE, 5, 0, 701)
    assert placebo.atk[(1, placebo.AWAY)] > placebo.atk[(1, placebo.HOME)]


def test_venue_ratings_are_seeded_symmetrically():
    """A promoted club has no reason to arrive with a venue asymmetry - if it
    did, the arm would be measuring its own seed."""
    venue = _found_league(VenueAttackDefence())
    venue.means.roll_season(LEAGUE)
    venue._seed_slots(99, LEAGUE)
    assert venue.atk[(99, venue.HOME)] == venue.atk[(99, venue.AWAY)]
    assert venue.dfn[(99, venue.HOME)] == venue.dfn[(99, venue.AWAY)]


# --- the replay ordering ----------------------------------------------------

def test_replay_never_lets_a_row_see_its_own_result():
    """test_ratings_lib.py's twin, over this phase's driver: features() before
    update(), checked where it can actually fail."""
    from build_venue_features import build

    def index(final_score):
        rows = [
            (100 + i, 8, 1, pd.Timestamp("2020-08-01") + pd.Timedelta(i, "D"),
             1 + (i % 4), 1 + ((i + 1) % 4), 1, 0)
            for i in range(12)
        ]
        rows.append((999, 8, 1, pd.Timestamp("2020-09-01"), 1, 2, *final_score))
        return pd.DataFrame(rows, columns=[
            "fixture_id", "league_id", "season_id", "date",
            "home_id", "away_id", "home_goals", "away_goals",
        ])

    pd.testing.assert_frame_equal(build(index((0, 0))), build(index((7, 0))))
