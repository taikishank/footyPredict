"""Unit tests for goals_lib.

Three things here are easy to get quietly wrong and impossible to spot in an
ablation table, which is the same reason test_ratings_lib.py exists:

  the orientation of the score matrix - transposing it swaps home and away
  and still produces a plausible-looking log loss;

  the analytic gradient - a wrong one does not crash, it just converges
  somewhere else, and the fit would look merely "a bit worse than the online
  rating" rather than broken;

  the time weighting - if it were inverted the model would be fitted on the
  distant past, and over a 26-season archive that is still a sane-looking
  number.

Run:
    ./inference-py/bin/python -m pytest ml/test_goals_lib.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import goals_lib as gl


def poisson(k: int, rate: float) -> float:
    return math.exp(-rate) * rate ** k / math.factorial(k)


def synthetic(n_teams=20, n_rounds=12, seed=0, rho=-0.10, hfa=0.22, drift=0.0):
    """A league generated from known attack/defence parameters.

    Every club plays every other home and away each round; goals are drawn
    from the model's own rates, so a correct fit must recover the parameters
    that produced them. `drift` rotates the parameters over time, which is what
    the half-life tests need.
    """
    rng = np.random.default_rng(seed)
    attack = rng.normal(0, 0.25, n_teams)
    defence = rng.normal(0, 0.20, n_teams)
    attack -= attack.mean()
    defence -= defence.mean()

    rows = []
    for round_idx in range(n_rounds):
        for home in range(n_teams):
            for away in range(n_teams):
                if home == away:
                    continue
                shift = drift * round_idx * (1 if home < n_teams // 2 else -1)
                lam = math.exp(attack[home] + shift - defence[away] + hfa)
                mu = math.exp(attack[away] - defence[home] - shift)
                # Scorelines are drawn from the corrected matrix, not from two
                # independent Poissons, so `rho` is genuinely in the data and
                # the fit has something to recover rather than a zero to find.
                matrix = gl.score_matrix([lam], [mu], rho)[0]
                cell = rng.choice(matrix.size, p=matrix.ravel())
                rows.append((
                    (n_rounds - round_idx) * 30.0, home, away,
                    cell // matrix.shape[1], cell % matrix.shape[1],
                ))
    days, home_ids, away_ids, hg, ag = map(np.array, zip(*rows))
    matches = {"days_ago": days, "home_id": home_ids, "away_id": away_ids,
               "home_goals": hg, "away_goals": ag}
    return matches, attack, defence


# --- the shape -------------------------------------------------------------

def test_matrix_is_a_distribution():
    matrix = gl.score_matrix([1.5, 0.3, 4.0], [1.2, 2.0, 0.1], rho=-0.12)
    assert np.allclose(matrix.sum(axis=(1, 2)), 1.0)
    assert (matrix >= 0).all()


def test_rho_zero_is_independent_poisson():
    """The uncorrected matrix must be the outer product, cell for cell."""
    matrix = gl.score_matrix([1.53], [1.18])[0]
    for i in (0, 1, 2, 5):
        for j in (0, 1, 3):
            assert matrix[i, j] == pytest.approx(poisson(i, 1.53) * poisson(j, 1.18), rel=1e-6)


def test_matrix_orientation_is_home_by_away():
    """Rows are the home side. A heavy home favourite must put its mass below
    the diagonal, and transposing the matrix must swap the two win
    probabilities - the check that catches a silent transpose."""
    probs = gl.outcome_probs([3.0], [0.5])[0]
    assert probs[0] > 0.8 and probs[2] < 0.05
    mirrored = gl.outcome_probs([0.5], [3.0])[0]
    assert mirrored[2] == pytest.approx(probs[0])
    assert mirrored[1] == pytest.approx(probs[1])


def test_outcome_probs_sum_to_one_and_cover_every_cell():
    probs = gl.outcome_probs([1.5, 2.4], [1.2, 0.9], rho=-0.1)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_tau_cells():
    """The four corrected cells, against the paper's expressions."""
    lam, mu, rho = 1.4, 1.1, -0.13
    assert gl.tau(0, 0, lam, mu, rho) == pytest.approx(1 - lam * mu * rho)
    assert gl.tau(0, 1, lam, mu, rho) == pytest.approx(1 + lam * rho)
    assert gl.tau(1, 0, lam, mu, rho) == pytest.approx(1 + mu * rho)
    assert gl.tau(1, 1, lam, mu, rho) == pytest.approx(1 - rho)
    assert gl.tau(2, 1, lam, mu, rho) == pytest.approx(1.0)
    assert gl.tau(0, 2, lam, mu, rho) == pytest.approx(1.0)


def test_negative_rho_moves_mass_into_low_scoring_draws():
    """The whole reason the correction exists: 0-0 and 1-1 are underpredicted
    by independence, and the draw probability rises because of it."""
    plain = gl.score_matrix([1.4], [1.1])[0]
    corrected = gl.score_matrix([1.4], [1.1], rho=-0.13)[0]
    assert corrected[0, 0] > plain[0, 0]
    assert corrected[1, 1] > plain[1, 1]
    assert corrected[0, 1] < plain[0, 1]
    assert corrected[1, 0] < plain[1, 0]
    assert gl.outcome_probs([1.4], [1.1], rho=-0.13)[0, 1] > gl.outcome_probs([1.4], [1.1])[0, 1]


def test_btts_prob_matches_brute_force_matrix_sum():
    """btts_prob's inclusion-exclusion shortcut against summing the cells it
    is shorthand for: every cell where both sides scored."""
    lam, mu, rho = 1.5, 1.2, -0.1
    matrix = gl.score_matrix([lam], [mu], rho)[0]
    assert gl.btts_prob([lam], [mu], rho)[0] == pytest.approx(matrix[1:, 1:].sum())


def test_btts_prob_rho_zero_matches_independent_poisson():
    """Under independence, BTTS is just two complementary events."""
    lam, mu = 1.53, 1.18
    expected = (1 - poisson(0, lam)) * (1 - poisson(0, mu))
    assert gl.btts_prob([lam], [mu])[0] == pytest.approx(expected, rel=1e-6)


def test_btts_prob_is_bounded():
    probs = gl.btts_prob([0.3, 1.5, 4.0], [0.2, 1.1, 3.5], rho=-0.1)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_negative_rho_raises_btts_prob():
    """tau moves mass into 1-1 (BTTS) and out of 1-0/0-1 (not BTTS) while
    barely touching either marginal - the same mechanism
    test_negative_rho_moves_mass_into_low_scoring_draws reads off the draw
    probability, read here off BTTS instead."""
    plain = gl.btts_prob([1.4], [1.1])[0]
    corrected = gl.btts_prob([1.4], [1.1], rho=-0.13)[0]
    assert corrected > plain


def test_correction_barely_moves_the_marginals():
    """tau redistributes between four adjacent cells rather than changing how
    many goals are expected - so the fitted rates stay interpretable as goal
    rates whatever rho does."""
    for rho in (-0.15, -0.05, 0.08):
        matrix = gl.score_matrix([1.5], [1.2], rho=rho)[0]
        goals = np.arange(gl.MAX_GOALS + 1)
        assert (matrix.sum(axis=1) * goals).sum() == pytest.approx(1.5, abs=0.03)
        assert (matrix.sum(axis=0) * goals).sum() == pytest.approx(1.2, abs=0.03)


def test_truncation_is_negligible():
    """MAX_GOALS is a choice, and this is the check that it is not a live one."""
    lam, mu = 3.5, 3.0
    tail = 1.0 - sum(poisson(i, lam) for i in range(gl.MAX_GOALS + 1))
    assert tail < 1e-5


# --- the gradient ----------------------------------------------------------

def test_analytic_gradient_matches_finite_differences():
    """Every parameter family at once: attack, defence, home advantage, rho."""
    matches, _, _ = synthetic(n_teams=6, n_rounds=3, seed=3)
    weights = 0.5 ** (matches["days_ago"] / 365.0)
    n = 6
    args = (matches["home_id"], matches["away_id"],
            matches["home_goals"].astype(float), matches["away_goals"].astype(float),
            weights, n)

    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.2, 2 * n), [0.22, -0.09]])
    _, grad = gl._negative_log_likelihood(x, *args)

    step = 1e-6
    for i in range(len(x)):
        up, down = x.copy(), x.copy()
        up[i] += step
        down[i] -= step
        numeric = (gl._negative_log_likelihood(up, *args)[0]
                   - gl._negative_log_likelihood(down, *args)[0]) / (2 * step)
        assert grad[i] == pytest.approx(numeric, rel=2e-4, abs=1e-5)


# --- the fit ---------------------------------------------------------------

def test_fit_recovers_generative_parameters():
    """The test that would catch a wrong gradient, a wrong sign or a wrong
    likelihood: goals drawn from known parameters, and the fit has to find
    them back."""
    matches, attack, defence = synthetic(n_teams=20, n_rounds=12, seed=1)
    params = gl.fit_league(matches, half_life=10_000.0)

    fitted_atk = np.array([params.attack[params.index[t]] for t in range(20)])
    fitted_def = np.array([params.defence[params.index[t]] for t in range(20)])
    assert np.corrcoef(fitted_atk, attack)[0, 1] > 0.9
    assert np.corrcoef(fitted_def, defence)[0, 1] > 0.9
    assert params.hfa == pytest.approx(0.22, abs=0.05)
    assert params.rho == pytest.approx(-0.10, abs=0.06)


def test_attack_is_recentred_without_moving_a_single_rate():
    """_project's contract. The level is unidentified; the rates are not."""
    matches, _, _ = synthetic(n_teams=8, n_rounds=6, seed=2)
    params = gl.fit_league(matches, half_life=365.0)
    assert params.attack.mean() == pytest.approx(0.0, abs=1e-9)

    before = params.rates(0, 1)
    shifted = gl.LeagueParams(params.index, params.attack + 0.3,
                              params.defence + 0.3, params.hfa, params.rho)
    assert shifted.rates(0, 1) == pytest.approx(before)


def test_time_weighting_prefers_the_recent_past():
    """A club that was strong early and weak lately must rate below one that
    did the opposite - under a short half-life, and much less so under a long
    one. An inverted weight would reverse both."""
    matches, _, _ = synthetic(n_teams=10, n_rounds=10, seed=4, drift=0.06)
    # `drift` raises the first half of the clubs' attack over time and lowers
    # the rest, so club 0 improves and club 9 declines.
    short = gl.fit_league(matches, half_life=60.0)
    long = gl.fit_league(matches, half_life=5_000.0)

    def gap(params):
        return (params.attack[params.index[0]] - params.attack[params.index[9]])

    assert gap(short) > gap(long)


def test_weight_floor_drops_the_distant_past():
    """A club seen only in the decayed past must leave the fit entirely - the
    property the cadence sweep's cost depends on."""
    matches, _, _ = synthetic(n_teams=6, n_rounds=4, seed=5)
    for key, value in (("days_ago", 9_000.0), ("home_id", 99), ("away_id", 0),
                       ("home_goals", 3), ("away_goals", 1)):
        matches[key] = np.append(matches[key], value)
    params = gl.fit_league(matches, half_life=100.0)
    assert 99 not in params.index
    assert 0 in params.index


def test_unseen_club_is_seeded_below_the_league():
    """A promoted side has no parameters and must not be handed the mean."""
    matches, _, _ = synthetic(n_teams=8, n_rounds=6, seed=6)
    params = gl.fit_league(matches, half_life=365.0)
    assert params._get(4242) == (-gl.PROMOTED_OFFSET, -gl.PROMOTED_OFFSET)
    # Against the same home side, the promoted club concedes more and scores
    # less than a club of exactly league-average strength would.
    average = gl.LeagueParams({**params.index, 4242: len(params.attack)},
                              np.append(params.attack, 0.0),
                              np.append(params.defence, 0.0),
                              params.hfa, params.rho)
    lam_vs_promoted, mu_of_promoted = params.rates(0, 4242)
    lam_vs_average, mu_of_average = average.rates(0, 4242)
    assert lam_vs_promoted > lam_vs_average
    assert mu_of_promoted < mu_of_average


def test_warm_start_reaches_the_same_optimum():
    """Warm-starting is a cost optimisation, not a modelling choice, so it has
    to land in the same place a cold fit does."""
    matches, _, _ = synthetic(n_teams=12, n_rounds=8, seed=7)
    cold = gl.fit_league(matches, half_life=365.0)
    stale = gl.LeagueParams(cold.index, cold.attack + 0.15, cold.defence - 0.1,
                            0.05, -0.02)
    warm = gl.fit_league(matches, half_life=365.0, warm=stale)
    assert np.allclose(warm.attack, cold.attack, atol=0.02)
    assert np.allclose(warm.defence, cold.defence, atol=0.02)
    assert warm.rho == pytest.approx(cold.rho, abs=0.02)


def test_thin_evidence_is_pulled_toward_the_league(monkeypatch):
    """The ridge's whole job: a club with one freak result must not be fitted
    from that result as confidently as a club with a season behind it.

    Measured against the same fit with the prior switched off rather than
    against a threshold, so the test says what the ridge *does* rather than
    asserting a number that moves whenever RIDGE is retuned.
    """
    matches, _, _ = synthetic(n_teams=10, n_rounds=8, seed=8)
    for key, value in (("days_ago", 1.0), ("home_id", 77), ("away_id", 0),
                       ("home_goals", 7), ("away_goals", 0)):
        matches[key] = np.append(matches[key], value)

    ridged = gl.fit_league(matches, half_life=365.0)
    monkeypatch.setattr(gl, "RIDGE", 0.0)
    unridged = gl.fit_league(matches, half_life=365.0)

    newcomer = ridged.attack[ridged.index[77]]
    assert newcomer < unridged.attack[unridged.index[77]] / 1.5

    # And the incumbents, who have a hundred-odd matches each, are barely
    # touched. Compared as *gaps* rather than levels: the level is unidentified
    # and _project pins it to a mean that includes the newcomer, whose value is
    # exactly what differs between these two fits.
    def gaps(params):
        levels = np.array([params.attack[params.index[t]] for t in range(10)])
        return levels - levels[0]

    assert np.allclose(gaps(ridged), gaps(unridged), atol=0.05)
