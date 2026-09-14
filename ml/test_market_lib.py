import pytest

from market_lib import (
    DEVIG_METHODS,
    devig_multiplicative,
    devig_shin,
    implied,
    learn_name_map,
    overround,
)

# A typical Pinnacle three-way closing line: a clear but not extreme home
# favourite, carrying about the 2.5% margin build_market_index.py --diagnose
# measures across the archive.
TYPICAL = (1.85, 3.70, 4.40)

# A heavy favourite, where the favourite-longshot effect the two methods
# disagree about is largest.
LOPSIDED = (1.15, 8.00, 19.00)


# --- what any de-vig has to do ---------------------------------------------

@pytest.mark.parametrize("method", DEVIG_METHODS.values())
@pytest.mark.parametrize("odds", [TYPICAL, LOPSIDED, (3.0, 3.0, 3.0)])
def test_result_is_a_distribution(method, odds):
    probs = method(odds)
    assert len(probs) == 3
    assert all(0.0 < p < 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0)


@pytest.mark.parametrize("method", DEVIG_METHODS.values())
def test_a_book_with_no_margin_is_left_alone(method):
    """Prices that already imply a distribution must survive de-vigging
    unchanged - there is nothing to remove."""
    fair = (1 / 0.5, 1 / 0.25, 1 / 0.25)
    assert method(fair) == pytest.approx((0.5, 0.25, 0.25))


@pytest.mark.parametrize("method", DEVIG_METHODS.values())
def test_the_favourite_stays_the_favourite(method):
    """De-vigging redistributes the margin; it must not reorder the outcomes."""
    raw = implied(LOPSIDED)
    probs = method(LOPSIDED)
    assert sorted(range(3), key=lambda i: raw[i]) == sorted(
        range(3), key=lambda i: probs[i]
    )


def test_overround_is_the_margin():
    assert overround((2.0, 4.0, 4.0)) == pytest.approx(1.0)
    assert overround(TYPICAL) > 1.0


# --- where the two methods disagree, and why it is worth having both -------

def test_shin_shades_the_favourite_up_relative_to_proportional():
    """The reason devig_shin exists.

    A bookmaker loads proportionally more margin onto longshots than onto
    favourites, so dividing through by the sum (the multiplicative method)
    leaves the favourite understated. Shin's correction has to push the
    favourite's probability *up* relative to multiplicative, and the longshot
    down. If this ever reverses, the implementation is wrong in the exact
    direction that would bias a 1X2 benchmark on heavy-favourite fixtures.
    """
    shin = devig_shin(LOPSIDED)
    proportional = devig_multiplicative(LOPSIDED)
    assert shin[0] > proportional[0]   # the favourite
    assert shin[2] < proportional[2]   # the longshot


def test_the_two_agree_when_the_book_is_balanced():
    """With three equal prices there is no favourite-longshot asymmetry to
    correct, so the methods must coincide."""
    assert devig_shin((3.0, 3.0, 3.0)) == pytest.approx(
        devig_multiplicative((3.0, 3.0, 3.0))
    )


def test_an_unreasonably_large_margin_falls_back_rather_than_extrapolating():
    """Shin's z is bracketed deliberately. A price that implies a margin
    outside the bracket is not something to extrapolate into - z near 1 is
    not a bookmaker's book - so it takes the proportional answer instead of
    returning a silently wrong one."""
    absurd = (1.2, 1.2, 1.2)  # implies ~2.5, a 150% margin
    assert devig_shin(absurd) == pytest.approx(devig_multiplicative(absurd))


# --- the learned name mapping ----------------------------------------------

def test_a_club_is_mapped_to_the_name_it_usually_matches():
    pairs = [(8, "West Bromwich Albion", "West Brom")] * 9
    pairs += [(8, "West Bromwich Albion", "Wolves")]  # one bad match
    mapping, ambiguous = learn_name_map(pairs)
    assert mapping[(8, "West Bromwich Albion")] == "West Brom"
    assert ambiguous == []


def test_a_split_vote_is_reported_rather_than_guessed():
    """A club whose votes are split means the score-and-date pass paired some
    fixtures wrongly. Taking the plurality would bake that error into every
    season the club appears in, so it is surfaced instead."""
    pairs = [(8, "Ambiguous FC", "Alpha")] * 5 + [(8, "Ambiguous FC", "Beta")] * 5
    mapping, ambiguous = learn_name_map(pairs)
    assert (8, "Ambiguous FC") not in mapping
    assert len(ambiguous) == 1
    assert ambiguous[0][0] == (8, "Ambiguous FC")


def test_the_same_name_in_two_leagues_maps_independently():
    """The archive's five leagues are disconnected components
    (build_rating_features.py --diagnose), so nothing stops two of them
    holding a club with the same name."""
    pairs = [(8, "Valencia", "Valencia EN")] * 5 + [(564, "Valencia", "Valencia ES")] * 5
    mapping, _ = learn_name_map(pairs)
    assert mapping[(8, "Valencia")] == "Valencia EN"
    assert mapping[(564, "Valencia")] == "Valencia ES"
