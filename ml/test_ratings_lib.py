import pytest

from ratings_lib import (
    ARMS,
    HOME_ADVANTAGE,
    INIT_RATING,
    PROMOTED_OFFSET,
    SEASON_REGRESSION,
    AttackDefence,
    EloRating,
    GoalDiffElo,
)

LEAGUE = 8


def _found_league(rating, team_ids=(1, 2, 3, 4)):
    """Play a round robin so the league has a settled cohort, the way a real
    season does before anyone is promoted into it."""
    for home in team_ids:
        for away in team_ids:
            if home != away:
                rating.update(LEAGUE, home, away, 1, 1) if False else rating.update(
                    home, away, LEAGUE, 1, 1
                )
    return rating


# --- the invariant every arm has to hold ------------------------------------

@pytest.mark.parametrize("name", ARMS)
def test_another_fixture_does_not_move_these_clubs_ratings(name):
    """A result between two other clubs must not touch this pair's own
    ratings.

    Stated over the club ratings rather than over features(), because
    AttackDefence's expected-goals columns legitimately *do* move: they also
    carry the league's baseline scoring rate and home advantage, which learn
    from every fixture in the league. That is not leakage - the replay is
    strictly in date order, so those two only ever see earlier matches - but
    it does mean this pair's rates are not a function of this pair alone.
    """
    rating = ARMS[name]()
    _found_league(rating)
    own = ("home_atk", "home_def", "away_atk", "away_def") if name == "atkdef" \
        else (rating.FIELD_ORDER[0], rating.FIELD_ORDER[1])
    before = rating.features(1, 2, LEAGUE)
    rating.update(3, 4, LEAGUE, 5, 0)  # a different fixture entirely
    after = rating.features(1, 2, LEAGUE)
    assert [before[c] for c in own] == [after[c] for c in own]


@pytest.mark.parametrize("name", ARMS)
def test_field_order_matches_emitted_keys(name):
    """FIELD_ORDER is the contract a hand-built inference vector relies on, the
    same way feature_lib.FORM_FIELD_ORDER is."""
    rating = ARMS[name]()
    assert tuple(rating.features(1, 2, LEAGUE)) == rating.FIELD_ORDER


# --- Elo --------------------------------------------------------------------

def test_elo_update_is_zero_sum():
    elo = _found_league(EloRating())
    total_before = sum(elo.ratings.values())
    elo.update(1, 2, LEAGUE, 3, 0)
    assert sum(elo.ratings.values()) == pytest.approx(total_before)


def test_beating_a_stronger_side_gains_more_than_beating_a_weaker_one():
    elo = _found_league(EloRating())
    elo.ratings.update({1: 1500.0, 2: 1800.0, 3: 1200.0})

    strong = EloRating()
    strong.ratings = dict(elo.ratings)
    strong.means = elo.means
    strong.update(1, 2, LEAGUE, 1, 0)

    weak = EloRating()
    weak.ratings = dict(elo.ratings)
    weak.means = elo.means
    weak.update(1, 3, LEAGUE, 1, 0)

    # The whole point of an opponent-strength rating: the same 1-0 is worth
    # more against the better side.
    assert strong.ratings[1] - 1500.0 > weak.ratings[1] - 1500.0


def test_home_advantage_makes_a_home_draw_cost_the_home_side():
    elo = _found_league(EloRating())
    elo.ratings.update({1: 1500.0, 2: 1500.0})
    elo.update(1, 2, LEAGUE, 1, 1)
    # Level sides, but the home team was expected to win, so a draw is a loss
    # of rating - which is what HOME_ADVANTAGE being inside the expectation
    # rather than stored on the rating means.
    assert HOME_ADVANTAGE > 0
    assert elo.ratings[1] < 1500.0
    assert elo.ratings[2] > 1500.0


def test_newcomers_are_seeded_below_the_league_they_join():
    elo = _found_league(EloRating())
    mean = elo.means.mean(LEAGUE, elo.ratings)
    elo.means.roll_season(LEAGUE)  # the cohort finishes its season
    promoted = elo.features(1, 99, LEAGUE)["away_elo"]
    assert promoted == pytest.approx(mean - PROMOTED_OFFSET)


def test_founding_cohort_is_not_treated_as_promoted():
    elo = EloRating()
    assert elo.features(1, 2, LEAGUE)["home_elo"] == INIT_RATING


def test_season_regression_scales_with_seasons_missed():
    elo = _found_league(EloRating())
    elo.means.roll_season(LEAGUE)
    target = elo.means.mean(LEAGUE, elo.ratings)

    one, four = EloRating(), EloRating()
    for rating in (one, four):
        rating.ratings = dict(elo.ratings)
        rating.means = elo.means
        rating.ratings[1] = target + 400.0
    # Recomputed after the club is moved, since season_boundary() reads the
    # league mean as it stands at the boundary - which includes this club.
    target = one.means.mean(LEAGUE, one.ratings)
    gap = one.ratings[1] - target
    one.season_boundary(LEAGUE, 1, seasons_elapsed=1)
    four.season_boundary(LEAGUE, 1, seasons_elapsed=4)

    assert one.ratings[1] == pytest.approx(target + gap * (1 - SEASON_REGRESSION))
    # A club four years out of the division is nearly back at the league mean,
    # which is the right prior for a side nobody in this archive has seen.
    assert abs(four.ratings[1] - target) < abs(one.ratings[1] - target)


def test_recenter_preserves_every_rating_difference():
    """recenter() exists to stop the level columns becoming a date proxy; it
    must be a no-op for anything the model can actually use."""
    elo = _found_league(EloRating())
    elo.ratings.update({1: 1400.0, 2: 1600.0, 3: 1500.0, 4: 1500.0})
    gaps = {(a, b): elo.ratings[a] - elo.ratings[b] for a in (1, 2) for b in (3, 4)}

    elo.recenter(LEAGUE)

    assert sum(elo.ratings[t] for t in (1, 2, 3, 4)) / 4 == pytest.approx(INIT_RATING)
    for pair, gap in gaps.items():
        assert elo.ratings[pair[0]] - elo.ratings[pair[1]] == pytest.approx(gap)


# --- goal-difference Elo ----------------------------------------------------

def test_margin_multiplier_is_increasing_and_concave():
    gd = GoalDiffElo()
    multipliers = [gd._multiplier(m, 0) for m in range(1, 7)]
    assert multipliers == sorted(multipliers)
    steps = [b - a for a, b in zip(multipliers, multipliers[1:])]
    # The jump from winning by one to winning by two says more about a side
    # than the jump from four to five, so the steps must shrink.
    assert steps[0] > steps[-1]


def test_goal_difference_elo_separates_a_rout_from_a_narrow_win():
    narrow, rout = _found_league(GoalDiffElo()), _found_league(GoalDiffElo())
    narrow.update(1, 2, LEAGUE, 1, 0)
    rout.update(1, 2, LEAGUE, 5, 0)
    assert rout.ratings[1] > narrow.ratings[1]


def test_plain_elo_does_not():
    narrow, rout = _found_league(EloRating()), _found_league(EloRating())
    narrow.update(1, 2, LEAGUE, 1, 0)
    rout.update(1, 2, LEAGUE, 5, 0)
    assert rout.ratings[1] == pytest.approx(narrow.ratings[1])


# --- attack / defence -------------------------------------------------------

def test_conceding_fewer_than_expected_raises_a_defence_rating():
    ad = _found_league(AttackDefence())
    before = ad.dfn[1]
    ad.update(1, 2, LEAGUE, 0, 0)  # a clean sheet against a baseline rate > 0
    assert ad.dfn[1] > before


def test_scoring_more_than_expected_raises_an_attack_rating():
    ad = _found_league(AttackDefence())
    before = ad.atk[1]
    ad.update(1, 2, LEAGUE, 6, 0)
    assert ad.atk[1] > before


def test_expected_goals_stay_finite_under_a_runaway_streak():
    ad = _found_league(AttackDefence())
    for _ in range(500):
        ad.update(1, 2, LEAGUE, 9, 0)
    feats = ad.features(1, 2, LEAGUE)
    assert feats["exp_goals_home"] <= AttackDefence.MAX_LAMBDA
    assert abs(feats["home_atk"]) <= AttackDefence.MAX_RATING


def test_recenter_preserves_expected_goals():
    ad = _found_league(AttackDefence())
    ad.update(1, 2, LEAGUE, 4, 0)
    ad.update(3, 4, LEAGUE, 0, 2)
    before = ad.features(1, 2, LEAGUE)
    ad.recenter(LEAGUE)
    after = ad.features(1, 2, LEAGUE)
    # Levels shift; the rates the model actually uses must not.
    assert after["exp_goal_diff"] == pytest.approx(before["exp_goal_diff"], abs=1e-9)


# --- the replay itself ------------------------------------------------------

def test_replay_never_lets_a_row_see_its_own_result():
    """The invariant the whole experiment rests on, tested where it can
    actually fail: in the driver's ordering, not in a rating's API.

    Replays the same fixtures twice, changing only the *last* score, and
    asserts every emitted row is identical - including that last fixture's own
    row. A driver that called update() before features() would leak here and
    nowhere else.
    """
    import pandas as pd

    from build_rating_features import build

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

    drawn = build(index((0, 0)))
    thrashed = build(index((7, 0)))
    pd.testing.assert_frame_equal(drawn, thrashed)
