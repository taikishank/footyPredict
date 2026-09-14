"""Turning bookmaker prices into a probability forecast, and matching a
price source's fixtures onto SportMonks fixture_ids.

Split out from build_market_index.py the same way ratings_lib.py is split
from build_rating_features.py: everything here is a pure function over
in-memory values, so the parts that are easy to get quietly wrong - the
de-vigging, and the club-name mapping - are unit-testable without touching
the network or the archive.

Two things happen here.

**De-vigging.** A bookmaker's three prices imply probabilities that sum to
more than one; the excess is the margin (the "overround" or "vig"). Comparing
a model against raw implied probabilities would score the market against a
distribution that isn't one, and would flatter the model by roughly the
margin. Two removal methods are implemented rather than one, because they
disagree in a direction that matters here - see devig_shin below.

**Name matching**, which is deliberately *learned* rather than hand-written.
See learn_name_map.
"""
from __future__ import annotations

from collections import Counter, defaultdict

# Bisection bounds for Shin's z. Real three-way football margins put z well
# under 0.1; 0.25 is a ceiling no sane price reaches, and is here only so the
# solver has a bracket rather than to express a belief about the data.
_Z_MAX = 0.25
_Z_TOLERANCE = 1e-10
_Z_ITERATIONS = 60


def implied(odds: tuple[float, ...]) -> tuple[float, ...]:
    """Raw implied probabilities, 1/price each. These sum to >1 by the margin."""
    return tuple(1.0 / o for o in odds)


def overround(odds: tuple[float, ...]) -> float:
    """How much more than 1.0 the raw implied probabilities sum to."""
    return sum(implied(odds))


def devig_multiplicative(odds: tuple[float, ...]) -> tuple[float, ...]:
    """Divide the raw implied probabilities by their sum.

    The standard first answer, and the one that assumes the margin is applied
    as a constant *proportional* loading across all three outcomes. Cheap,
    and unbiased if that assumption holds.
    """
    raw = implied(odds)
    total = sum(raw)
    return tuple(p / total for p in raw)


def devig_shin(odds: tuple[float, ...]) -> tuple[float, ...]:
    """Shin's method: back out the margin assuming some insider betting.

    The proportional assumption above is known to be wrong in a specific
    direction - a bookmaker loads proportionally *more* margin onto longshots
    than onto favourites, so dividing through by the sum leaves longshots
    overstated and favourites understated. That bias lands squarely on the
    draw and on heavy-favourite fixtures, which is exactly where a 1X2 model
    is judged, so taking the cheap method on faith would put a thumb on the
    scale of the benchmark this whole module exists to produce.

    Shin models the margin as the bookmaker's defence against a fraction `z`
    of insider money, which yields, for each outcome:

        p_i = [sqrt(z^2 + 4(1 - z) * raw_i^2 / S) - z] / (2 * (1 - z))

    where raw_i = 1/odds_i and S = sum(raw). `z` is whatever makes the p_i
    sum to one, found here by bisection - the expression is monotone
    decreasing in z over [0, _Z_MAX), so bisection is sufficient and has no
    convergence failure mode worth handling.

    Reported alongside the multiplicative numbers rather than instead of
    them: if the two agree, the de-vig choice is not load-bearing and the
    benchmark can be quoted without a caveat.
    """
    raw = implied(odds)
    total = sum(raw)

    def summed(z: float) -> float:
        if z <= 0.0:
            return total  # the z -> 0 limit is the raw sum, unnormalised
        return sum(
            ((z * z + 4.0 * (1.0 - z) * r * r / total) ** 0.5 - z) / (2.0 * (1.0 - z))
            for r in raw
        )

    lo, hi = 0.0, _Z_MAX
    if summed(hi) > 1.0:
        # The margin is larger than the bracket can explain. Falling back is
        # safer than extending the bracket, since z near 1 is not a price.
        return devig_multiplicative(odds)

    for _ in range(_Z_ITERATIONS):
        mid = (lo + hi) / 2.0
        if summed(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _Z_TOLERANCE:
            break

    z = (lo + hi) / 2.0
    if z <= 0.0:
        return devig_multiplicative(odds)
    probs = tuple(
        ((z * z + 4.0 * (1.0 - z) * r * r / total) ** 0.5 - z) / (2.0 * (1.0 - z))
        for r in raw
    )
    # Bisection leaves a residual well under 1e-9; normalising makes the
    # output exactly a distribution so callers never have to think about it.
    s = sum(probs)
    return tuple(p / s for p in probs)


DEVIG_METHODS = {
    "multiplicative": devig_multiplicative,
    "shin": devig_shin,
}


# --- club-name matching ----------------------------------------------------
#
# The two sources spell clubs differently ("West Bromwich Albion" vs "West
# Brom", "FC Bayern München" vs "Bayern Munich", "Olympique Lyonnais" vs
# "Lyon"). The ingestor solves the live version of this with a token-overlap
# heuristic plus a hand-maintained alias table
# (ingestor-go/internal/odds/match.go), which is the right shape there: it
# has to match an event it has never seen before, right now.
#
# Offline it is the wrong shape, for two reasons. Fuzzy scoring reaches about
# 79% of fixtures here and the misses are not random - they concentrate on
# clubs whose two spellings share no tokens at all, which is to say on
# specific clubs, in specific leagues, across every season they appear. That
# turns a name-matching gap into a *systematically missing subset of the
# benchmark*, which is precisely the failure the rating phase's --diagnose
# checks exist to catch. And an alias table is a list of guesses that has to
# be maintained by hand forever.
#
# The archive offers something the live path does not: both sources already
# agree on the answer. A fixture is pinned down by its league, its date and
# its scoreline without consulting a single name, and where that triple is
# unique it identifies the match outright. Those unambiguous matches then
# *say* what each club is called in the other source, so the mapping is read
# off the data instead of being asserted.


def learn_name_map(
    pairs: list[tuple[int, str, str]], dominance: float = 0.7
) -> tuple[dict[tuple[int, str], str], list[tuple[tuple[int, str], list]]]:
    """Read a (league, source name) -> target name mapping off matched fixtures.

    `pairs` is one (league_id, source_name, target_name) per club per
    unambiguously-matched fixture - so a club appearing in 300 matched
    fixtures votes 300 times, and a single mis-match cannot outvote it.

    A name is only accepted if it takes at least `dominance` of its club's
    votes. Anything short of that is returned in the second element rather
    than guessed at: a split vote means the score-and-date matching paired
    some fixtures wrongly, and silently taking the plurality would bake that
    error into every season the club appears in.

    Keyed by (league_id, name) rather than by name alone because the archive's
    five leagues are disconnected (build_rating_features.py --diagnose) and
    nothing guarantees two leagues don't each hold a club called "Valencia".
    """
    votes: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for league_id, source, target in pairs:
        votes[(league_id, source)][target] += 1

    mapping: dict[tuple[int, str], str] = {}
    ambiguous: list[tuple[tuple[int, str], list]] = []
    for key, counter in votes.items():
        top, n = counter.most_common(1)[0]
        if n / sum(counter.values()) >= dominance:
            mapping[key] = top
        else:
            ambiguous.append((key, counter.most_common(3)))
    return mapping, ambiguous
