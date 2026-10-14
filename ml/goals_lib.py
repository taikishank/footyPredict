"""Dixon-Coles: predict the scoreline, and read 1X2 off the score matrix.

Every model this project has fitted classifies the *outcome* directly. This
one does not. It models the two goal counts, and P(draw) is whatever the
diagonal of the resulting score matrix sums to - a structural consequence of
the two rates rather than a quantity anything was trained on.

That is the reason to try it. `FINDINGS_MARKET.md` put 35.5% of the gap to
the closing line on draws, and a classifier is poorly placed to fix that: the
draw is not one end of a strength axis, so no monotone function of a rating
gap picks it out. A score matrix gets it from the geometry instead.

Two separable pieces, and the phase measures them separately because they can
fail independently:

**The shape** - a score matrix over two Poisson rates, plus Dixon and Coles'
low-score correction `tau`. Independent Poisson understates 0-0 and 1-1; the
correction is four multiplicative cells and one parameter `rho`, and is the
"Dixon-Coles" part of Dixon-Coles.

**The fit** - attack and defence parameters recovered by a *time-weighted
batch maximum likelihood* over the recent past, refitted as the archive
advances. `ratings_lib.AttackDefence` already produces goal rates from the
same parameterisation, but it reaches them by one online gradient step per
fixture, tuned as a feature generator. This solves for them instead, which is
the version of the model Dixon and Coles actually describe.

The time weighting is the other half of their paper: a match `t` days old
enters the likelihood with weight 0.5 ** (t / HALF_LIFE), because a club's
strength two years ago is evidence about today but not equal evidence. The
half-life is swept rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

#: Scorelines beyond this are folded in by normalising the matrix. The rate
#: that matters is not the archive's average (1.53 home, 1.18 away) but
#: LAMBDA_CAP, the most extreme fixture the fit can produce: at 15 the missing
#: tail there is ~1e-4 of one margin and ~1e-6 at a realistic 3.5, both far
#: below the 1e-4 differences this phase is trying to measure.
MAX_GOALS = 15

#: `rho` bounds. Negative is the expected sign - the correction exists because
#: low-scoring draws are more common than independence predicts - and the
#: floor also keeps `tau` positive: the binding cell is 1 + lambda * rho, so
#: -0.18 is safe for any rate under 5.5 goals, which LAMBDA_CAP enforces.
RHO_MIN, RHO_MAX = -0.18, 0.10

#: Log-scale caps on the fitted parameters, the batch twin of
#: ratings_lib.AttackDefence.MAX_RATING. A club with three matches in the
#: weighted window can otherwise be handed an unbounded attack by a single
#: 5-0; the ridge below is the soft version of the same guard.
PARAM_CAP = 1.5
LAMBDA_CAP = 5.5

#: Prior precision pulling every club's attack and defence toward the league
#: mean. Not a regularisation nicety: the weighted window always contains a
#: promoted club with four matches in it, and the likelihood alone would fit
#: that club from four matches as confidently as an incumbent from ninety.
#:
#: Deliberately a *fixed* precision rather than one scaled by how little
#: evidence a club has. Scaling it that way double-counts: the likelihood's
#: own curvature for a club is already proportional to its match weight, so a
#: constant prior shrinks a four-match club hard and a full-season club barely,
#: which is the whole intent. 5.0 is worth roughly three or four matches of
#: evidence at the archive's scoring rate - a 7-0 on debut moves a club to
#: about a fifth of where the likelihood alone would put it.
RIDGE = 5.0

#: Matches whose time weight has decayed below this are dropped from the fit
#: entirely. At a 365-day half-life that is everything older than ~6.6 years,
#: which contributes 1% of a recent match's evidence and 100% of its cost -
#: and the cost is what decides whether a cadence sweep is affordable.
WEIGHT_FLOOR = 0.01

#: What a club nobody has seen is worth, on the log-goal scale, relative to
#: the league mean. Mirrors ratings_lib's PROMOTED_OFFSET, which
#: build_rating_features.py --diagnose measured rather than assumed: promoted
#: sides both score less and concede more, so one offset applies with the sign
#: each parameter needs.
PROMOTED_OFFSET = 0.10

#: Starting log home advantage, seeded at the level the archive settles on for
#: the same reason ratings_lib.AttackDefence.INIT_HFA is: fitted from zero,
#: the first seasons of the replay carry a systematically smaller home
#: advantage than the last, and a column that ramps across the archive is an
#: era proxy a tree will happily split on.
INIT_HFA = 0.20


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon and Coles' low-score correction, as a multiplier on the
    independent-Poisson probability of a scoreline.

    Independence is wrong in a specific, small place: 0-0 and 1-1 happen more
    often than two independent Poissons predict, and 1-0 and 0-1 less often.
    Only the four cells below are touched; every other scoreline keeps its
    independent probability, which is why one parameter is enough.

    The multipliers are constructed so that the correction leaves both
    *marginal* goal distributions almost unchanged - it moves mass between
    four adjacent cells rather than changing how many goals the model expects.
    """
    x, y = np.asarray(home_goals), np.asarray(away_goals)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def _poisson_pmf(rates: np.ndarray) -> np.ndarray:
    """P(k) for k in 0..MAX_GOALS, one row per rate.

    Written out rather than taken from scipy.stats so the replay does not pay
    scipy's per-call validation overhead thousands of times; the recurrence
    P(k) = P(k-1) * rate / k is also numerically kinder than exp(-r) r^k / k!
    at the tail.
    """
    rates = np.asarray(rates, dtype=float).reshape(-1, 1)
    k = np.arange(MAX_GOALS + 1)
    log_pmf = -rates + k * np.log(np.clip(rates, 1e-12, None)) - _LOG_FACTORIAL
    return np.exp(log_pmf)


_LOG_FACTORIAL = np.cumsum(np.concatenate([[0.0], np.log(np.arange(1, MAX_GOALS + 1))]))


def score_matrix(lam, mu, rho: float = 0.0) -> np.ndarray:
    """P(home scores i, away scores j) for every i, j up to MAX_GOALS.

    Shape (n, MAX_GOALS+1, MAX_GOALS+1), one matrix per fixture. Rows are the
    home side's goals. Normalised, so the truncated tail and any mass `tau`
    moves are both absorbed rather than left to bias the outcome sums.
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    mu = np.atleast_1d(np.asarray(mu, dtype=float))
    matrix = _poisson_pmf(lam)[:, :, None] * _poisson_pmf(mu)[:, None, :]
    if rho:
        cells = tau(
            np.array([[0, 0], [1, 1]])[None, :, :],     # home goals, 2x2
            np.array([[0, 1], [0, 1]])[None, :, :],     # away goals, 2x2
            lam[:, None, None], mu[:, None, None], rho,
        )
        matrix[:, :2, :2] *= np.clip(cells, 1e-9, None)
    return matrix / matrix.sum(axis=(1, 2), keepdims=True)


def outcome_probs(lam, mu, rho: float = 0.0) -> np.ndarray:
    """(n, 3) of P(home win), P(draw), P(away win), in train_v2.LABEL_MAP order.

    This is the whole point of the model: the draw probability is the matrix
    diagonal, so it is high when two rates are low and similar and low when
    they are not - a shape no monotone function of a rating gap can produce.
    """
    matrix = score_matrix(lam, mu, rho)
    draw = np.einsum("nii->n", matrix)
    upper = np.triu_indices(MAX_GOALS + 1, k=1)
    home = matrix[:, upper[1], upper[0]].sum(axis=1)
    away = matrix[:, upper[0], upper[1]].sum(axis=1)
    return np.stack([home, draw, away], axis=1)


def btts_prob(lam, mu, rho: float = 0.0) -> np.ndarray:
    """(n,) of P(both teams score), read off the same matrix as outcome_probs.

    "Neither scores" is P(home=0) + P(away=0) - P(home=0, away=0), the
    inclusion-exclusion correction for the cell double-counted by both
    marginals. BTTS is everything else - no new fit, no new parameter, the
    same geometry outcome_probs already pays for.
    """
    matrix = score_matrix(lam, mu, rho)
    p_home_blank = matrix[:, 0, :].sum(axis=1)
    p_away_blank = matrix[:, :, 0].sum(axis=1)
    p_either_blank = p_home_blank + p_away_blank - matrix[:, 0, 0]
    return 1.0 - p_either_blank


@dataclass
class LeagueParams:
    """One league's fitted state, as of some date.

    `attack` and `defence` are log-scale offsets, both signed so that larger is
    better: a rate is exp(attack_scorer - defence_conceder + hfa), so a good
    defence subtracts. `index` maps team_id to a row, and is part of the state
    because warm-starting the next refit means carrying the parameters of
    clubs the new window may no longer contain.
    """

    index: dict[int, int]
    attack: np.ndarray
    defence: np.ndarray
    hfa: float
    rho: float

    def rates(self, home_id: int, away_id: int) -> tuple[float, float]:
        """Pre-match (lambda, mu) for one fixture, seeding unseen clubs."""
        atk_h, def_h = self._get(home_id)
        atk_a, def_a = self._get(away_id)
        lam = min(np.exp(atk_h - def_a + self.hfa), LAMBDA_CAP)
        mu = min(np.exp(atk_a - def_h), LAMBDA_CAP)
        return float(lam), float(mu)

    def _get(self, team_id: int) -> tuple[float, float]:
        row = self.index.get(team_id)
        if row is None:
            # Same rule as ratings_lib._seed: a club this fit has never seen is
            # a promoted one, and starts below the league mean rather than at
            # it. The parameters are mean-zero by construction (see _project),
            # so the mean is 0 and the offset is the whole seed.
            return -PROMOTED_OFFSET, -PROMOTED_OFFSET
        return float(self.attack[row]), float(self.defence[row])


def _project(attack: np.ndarray, defence: np.ndarray) -> None:
    """Re-centre attack on zero, in place, without moving a single rate.

    The likelihood cannot identify the level of these parameters: adding c to
    every attack and every defence leaves every lambda and every mu exactly
    where they were, because each rate is a difference of one of each. Left
    alone across thousands of warm-started refits that free direction wanders,
    and the emitted parameter columns start tracking the archive rather than
    the football - `ratings_lib.Rating.recenter`'s failure mode, arriving here
    by a different route. Shifting both by the same constant is exact.
    """
    shift = attack.mean()
    attack -= shift
    defence -= shift


def _negative_log_likelihood(x, home, away, hg, ag, weights, n_teams):
    """Weighted negative log-likelihood and its gradient.

    Analytic rather than numerical because of the cost structure: a league's
    fit carries ~2 x 24 + 2 parameters, so a finite-difference gradient is
    fifty likelihood evaluations per iteration, and the cadence sweep runs
    thousands of fits. The gradient is cheap to write because the Poisson part
    is the same "surprise" shape the online rating uses -
    d/d(log lambda) = x - lambda - and `tau` touches four cells.
    """
    attack, defence = x[:n_teams], x[n_teams:2 * n_teams]
    hfa, rho = x[2 * n_teams], x[2 * n_teams + 1]

    lam = np.exp(np.clip(attack[home] - defence[away] + hfa, None, np.log(LAMBDA_CAP)))
    mu = np.exp(np.clip(attack[away] - defence[home], None, np.log(LAMBDA_CAP)))

    t = tau(hg, ag, lam, mu, rho)
    t = np.clip(t, 1e-9, None)
    ll = weights * (hg * np.log(lam) - lam + ag * np.log(mu) - mu + np.log(t))

    # d(tau)/d(lambda), d(tau)/d(mu), d(tau)/d(rho), cell by cell.
    d_lam = np.where((hg == 0) & (ag == 0), -mu * rho, 0.0)
    d_lam += np.where((hg == 0) & (ag == 1), rho, 0.0)
    d_mu = np.where((hg == 0) & (ag == 0), -lam * rho, 0.0)
    d_mu += np.where((hg == 1) & (ag == 0), rho, 0.0)
    d_rho = np.where((hg == 0) & (ag == 0), -lam * mu, 0.0)
    d_rho += np.where((hg == 0) & (ag == 1), lam, 0.0)
    d_rho += np.where((hg == 1) & (ag == 0), mu, 0.0)
    d_rho += np.where((hg == 1) & (ag == 1), -1.0, 0.0)

    # Chain rule onto the log-scale parameters: d(lambda)/d(attack) = lambda.
    g_lam = weights * (hg - lam + lam * d_lam / t)
    g_mu = weights * (ag - mu + mu * d_mu / t)

    grad = np.zeros_like(x)
    np.add.at(grad, home, -g_lam)                       # attack_home
    np.add.at(grad, away, -g_mu)                        # attack_away
    np.add.at(grad, n_teams + away, g_lam)              # defence_away
    np.add.at(grad, n_teams + home, g_mu)               # defence_home
    grad[2 * n_teams] = -g_lam.sum()                    # hfa
    grad[2 * n_teams + 1] = -(weights * d_rho / t).sum()

    # Ridge toward the league mean. Both parameter families carry the same
    # prior; neither has a reason to be freer than the other.
    params = np.concatenate([attack, defence])
    grad[:2 * n_teams] += RIDGE * params
    ridge = 0.5 * RIDGE * (params ** 2).sum()

    return -ll.sum() + ridge, grad


def fit_league(matches, half_life: float, warm: LeagueParams | None = None,
               max_iter: int = 200) -> LeagueParams:
    """Time-weighted maximum-likelihood fit over one league's recent past.

    `matches` must contain only fixtures played strictly before the moment
    this fit is for - the caller owns that, and build_goals_features.py's
    replay is what enforces it. Columns: days_ago, home_id, away_id,
    home_goals, away_goals.

    Warm-starting from the previous refit is what makes a cadence sweep
    affordable: consecutive fits differ by days of football, so L-BFGS lands
    in a handful of iterations rather than a hundred.
    """
    weights = 0.5 ** (np.asarray(matches["days_ago"], dtype=float) / half_life)
    keep = weights >= WEIGHT_FLOOR
    weights = weights[keep]
    home_ids = np.asarray(matches["home_id"])[keep]
    away_ids = np.asarray(matches["away_id"])[keep]
    hg = np.asarray(matches["home_goals"], dtype=float)[keep]
    ag = np.asarray(matches["away_goals"], dtype=float)[keep]

    teams = np.unique(np.concatenate([home_ids, away_ids]))
    index = {int(t): i for i, t in enumerate(teams)}
    n = len(teams)
    home = np.array([index[int(t)] for t in home_ids])
    away = np.array([index[int(t)] for t in away_ids])

    x0 = np.zeros(2 * n + 2)
    x0[2 * n] = INIT_HFA
    if warm is not None:
        x0[2 * n] = warm.hfa
        x0[2 * n + 1] = warm.rho
        for team_id, row in index.items():
            old = warm.index.get(team_id)
            if old is not None:
                x0[row] = warm.attack[old]
                x0[n + row] = warm.defence[old]
            else:
                x0[row] = x0[n + row] = -PROMOTED_OFFSET

    bounds = ([(-PARAM_CAP, PARAM_CAP)] * (2 * n)
              + [(-1.0, 1.0), (RHO_MIN, RHO_MAX)])
    result = minimize(
        _negative_log_likelihood, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        args=(home, away, hg, ag, weights, n),
        options={"maxiter": max_iter},
    )
    attack, defence = result.x[:n].copy(), result.x[n:2 * n].copy()
    _project(attack, defence)
    return LeagueParams(index, attack, defence, float(result.x[2 * n]),
                        float(result.x[2 * n + 1]))
