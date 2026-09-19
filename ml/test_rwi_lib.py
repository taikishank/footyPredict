import math

import pytest

from ratings_lib import AttackDefence
from rwi_lib import LINEAR_SCALE, BoundedResidual, CoupledVenue, RWI_ARMS
from venue_lib import VenueAttackDefence

LEAGUE = 8


def _found_league(rating, team_ids=(1, 2, 3, 4)):
    """A round robin, so the league has a settled cohort before anyone is
    promoted into it - test_ratings_lib.py's helper, for the same reason."""
    for home in team_ids:
        for away in team_ids:
            if home != away:
                _update(rating, home, away, 1, 1)
    return rating


def _update(rating, home, away, hg, ag, fixture_id=1):
    extra = (fixture_id,) if isinstance(rating, VenueAttackDefence) else ()
    rating.update(home, away, LEAGUE, hg, ag, *extra)


def _features(rating, home, away, fixture_id=1):
    extra = (fixture_id,) if isinstance(rating, VenueAttackDefence) else ()
    return rating.features(home, away, LEAGUE, *extra)


# --- the contracts every arm has to hold ------------------------------------

@pytest.mark.parametrize("name", RWI_ARMS)
def test_field_order_matches_emitted_keys(name):
    rating = RWI_ARMS[name]()
    assert tuple(_features(rating, 1, 2)) == rating.FIELD_ORDER


@pytest.mark.parametrize("name", RWI_ARMS)
def test_expected_goals_stay_finite_under_a_runaway_streak(name):
    rating = _found_league(RWI_ARMS[name]())
    for i in range(500):
        _update(rating, 1, 2, 9, 0, fixture_id=i)
    values = _features(rating, 1, 2)
    assert all(math.isfinite(v) for v in values.values())
    assert values["exp_goals_home"] <= AttackDefence.MAX_LAMBDA


# --- the integrity arms: each must reproduce a published rating -------------

def test_bounded_lin_is_the_served_rating():
    """tanh at LINEAR_SCALE is the identity, so bounded_lin must be atkdef.

    This is what build_rwi_features.py asserts against ratings.csv over the
    whole archive; here it is the same claim on a handful of fixtures, so the
    unit tests catch a broken residual hook without a four-minute replay.
    """
    served, bounded = _found_league(AttackDefence()), _found_league(BoundedResidual(
        scale=LINEAR_SCALE, shape="tanh"))
    for i, (hg, ag) in enumerate([(3, 0), (0, 0), (1, 4), (7, 1), (2, 2)]):
        _update(served, 1, 2, hg, ag)
        _update(bounded, 1, 2, hg, ag)
    a, b = _features(served, 1, 2), _features(bounded, 1, 2)
    for field in AttackDefence.FIELD_ORDER:
        assert a[field] == pytest.approx(b[field], abs=1e-9)


def test_couple_c0_is_the_venue_rating():
    """CoupledVenue with the coupling off must be VenueAttackDefence."""
    venue, coupled = _found_league(VenueAttackDefence()), _found_league(
        CoupledVenue(coupling=0.0))
    for i, (hg, ag) in enumerate([(3, 0), (0, 0), (1, 4), (7, 1), (2, 2)]):
        _update(venue, 1, 2, hg, ag, fixture_id=i)
        _update(coupled, 1, 2, hg, ag, fixture_id=i)
    a, b = _features(venue, 1, 2), _features(coupled, 1, 2)
    for field in AttackDefence.FIELD_ORDER:
        assert a[field] == pytest.approx(b[field], abs=1e-12)


def test_the_residual_hook_leaves_the_served_rating_untouched():
    """AttackDefence._residual is the identity, so adding it changed nothing.

    The hook was added to ratings_lib for this phase; if it ever stops being
    the identity, every published number in the project moves.
    """
    rating = AttackDefence()
    for err in (-1.5, -0.2, 0.0, 0.7, 6.0, 9.0):
        assert rating._residual(err) == err


# --- BoundedResidual ---------------------------------------------------------

def test_tanh_compresses_the_tail_and_not_the_middle():
    rating = BoundedResidual(scale=2.0, shape="tanh")
    # Near zero it is linear to within a percent, which is the point: an
    # ordinary result updates as it does today.
    assert rating._residual(0.1) == pytest.approx(0.1, rel=0.01)
    # A 6-0 against a lambda of 1.5 is the case the arm exists for.
    assert rating._residual(4.5) < 2.0
    assert rating._residual(4.5) > rating._residual(2.0)


def test_clip_is_the_identity_inside_the_bound():
    rating = BoundedResidual(scale=2.0, shape="clip")
    assert rating._residual(1.9) == 1.9
    assert rating._residual(-1.9) == -1.9
    assert rating._residual(9.0) == 2.0
    assert rating._residual(-9.0) == -2.0


def test_a_blowout_moves_a_bounded_rating_less_than_the_served_one():
    """The mechanism, stated directly. Same fixtures, same learning rate."""
    served, bounded = _found_league(AttackDefence()), _found_league(
        BoundedResidual(scale=1.0, shape="tanh"))
    _update(served, 1, 2, 9, 0)
    _update(bounded, 1, 2, 9, 0)
    assert bounded.atk[1] < served.atk[1]


def test_unknown_shape_and_bad_scale_are_rejected():
    with pytest.raises(AssertionError):
        BoundedResidual(shape="sigmoid")
    with pytest.raises(AssertionError):
        BoundedResidual(scale=0.0)


# --- CoupledVenue ------------------------------------------------------------

def test_coupling_shrinks_the_gap_between_a_clubs_two_slots():
    """The sweep's axis, checked at both ends and in the middle."""
    gaps = {}
    for coupling in (0.0, 0.25, 0.75):
        rating = _found_league(CoupledVenue(coupling=coupling))
        # Club 1 is strong at home and poor away, so the two slots separate.
        for i in range(40):
            _update(rating, 1, 2, 4, 0, fixture_id=i)
            _update(rating, 3, 1, 4, 0, fixture_id=i + 1000)
        gaps[coupling] = abs(rating.atk[(1, rating.HOME)]
                             - rating.atk[(1, rating.AWAY)])
    assert gaps[0.0] > gaps[0.25] > gaps[0.75]


def test_full_coupling_collapses_the_two_slots():
    rating = _found_league(CoupledVenue(coupling=1.0))
    for i in range(20):
        _update(rating, 1, 2, 4, 0, fixture_id=i)
    assert rating.atk[(1, rating.HOME)] == pytest.approx(
        rating.atk[(1, rating.AWAY)], abs=1e-12)
    assert rating.dfn[(1, rating.HOME)] == pytest.approx(
        rating.dfn[(1, rating.AWAY)], abs=1e-12)


def test_coupling_pulls_both_slots_not_only_the_one_that_played():
    """Docstring claim: the shrink is club-level, so a home match moves the
    away slot too. Without that the arm is a decay, not a shared state."""
    rating = _found_league(CoupledVenue(coupling=0.5))
    before = rating.atk[(1, rating.AWAY)]
    _update(rating, 1, 2, 5, 0)
    assert rating.atk[(1, rating.AWAY)] != before


def test_coupling_out_of_range_is_rejected():
    with pytest.raises(AssertionError):
        CoupledVenue(coupling=1.5)
    with pytest.raises(AssertionError):
        CoupledVenue(coupling=-0.1)
