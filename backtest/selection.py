"""How a set of probabilities becomes a bet - the selection rule, named and
kept separate from the model that produced the probabilities.

These were previously two hardcoded expressions inside grade_calls.py, which
made "which side do we back" a property of the grader rather than something
you could vary and measure. They are different axes: `model_version` says
which artifact produced model_home/draw/away, `call_rule` says what we did
with them. A fixture can be graded once per (model_version, call_rule), and
because both rows grade the SAME edges snapshot, any difference between them
is selection and nothing else - the probabilities are identical by
construction rather than by being reconstructed twice and hoped over.

Why this axis exists at all: the three edges sum to exactly 0 (the model's
probabilities sum to 1, and the market's are de-vigged to sum to 1 in
ingestor-go/internal/odds/parse.go), so max_edge's pick is non-negative on
every fixture - measured 777/777 on v3. "The model found an edge here" is
therefore not information, and worse, the pick is close to uncorrelated with
what the model actually believes: on the same 777 snapshots argmax(edge)
agreed with argmax(model prob) on 147 (18.9%), and 387 of the picks were the
draw, an outcome v3 ranks first exactly 0 times. The median model
probability of the side max_edge backs is 0.294.

That is a property of the rule, not of v3: any model whose draw probabilities
sit above the de-vigged market's will be steered into draws by max_edge, and
v3's do on 96.8% of snapshots (median +3.1pp). max_prob is the control that
separates the two explanations.
"""
from __future__ import annotations

from typing import Callable

OUTCOMES = ("home_win", "draw", "away_win")

#: The rule every row written before this module existed was following.
#: grade_calls.py and the /fixtures/upcoming UI both hardcoded it.
DEFAULT_RULE = "max_edge"

Probs = dict[str, float]


def _max_edge(model: Probs, market: Probs) -> str:
    """The outcome the model thinks the market most UNDERprices.

    Max SIGNED edge, not max |edge|: the largest disagreement in absolute
    terms is frequently one the model thinks is OVERpriced, and backing that
    is the opposite of the strategy. Never refuses - the edges sum to 0, so
    at least one is always non-negative.
    """
    return max(OUTCOMES, key=lambda o: model[o] - market[o])


def _max_prob(model: Probs, market: Probs) -> str:
    """The outcome the model thinks is most likely, market ignored.

    The belief arm. Names a side on every fixture too, so it refuses exactly
    as often as max_edge does (never) and the two are comparable on the full
    book rather than on differently-sized slices.
    """
    return max(OUTCOMES, key=lambda o: model[o])


#: rule name -> picks the single outcome to back.
SINGLE: dict[str, Callable[[Probs, Probs], str]] = {
    "max_edge": _max_edge,
    "max_prob": _max_prob,
}


def _dc_exclude_max_edge(model: Probs, market: Probs) -> str:
    """Double-chance under max_edge: exclude the outcome the model thinks is
    most OVERpriced, i.e. min signed edge.

    The DC label excluding outcome `o` has edge (1 - model[o]) - (1 - market[o])
    == -edge[o], so maximising DC edge is minimising single edge. This is
    deliberately NOT derived from the singles pick: while it was ("the pair
    excluding call_side"), dc_correct was the exact negation of correct and
    the arm measured nothing the singles arm had not.
    """
    return min(OUTCOMES, key=lambda o: model[o] - market[o])


def _dc_exclude_max_prob(model: Probs, market: Probs) -> str:
    """Double-chance under max_prob: exclude the outcome the model thinks is
    least likely, backing the two it believes in. The belief analogue, and
    like its singles counterpart it never consults the market."""
    return min(OUTCOMES, key=lambda o: model[o])


#: rule name -> picks the outcome the double-chance bet EXCLUDES.
DOUBLE_CHANCE: dict[str, Callable[[Probs, Probs], str]] = {
    "max_edge": _dc_exclude_max_edge,
    "max_prob": _dc_exclude_max_prob,
}

BTTS_OUTCOMES = ("yes", "no")


def _max_edge_btts(model: Probs, market: Probs) -> str:
    """BTTS's own max_edge, over ("yes", "no") rather than OUTCOMES - a
    native two-outcome pick, not DOUBLE_CHANCE's "exclude one of three"
    shape forced down to two. Same max-signed-edge argument as _max_edge:
    the two edges sum to 0, so this never refuses either."""
    return max(BTTS_OUTCOMES, key=lambda o: model[o] - market[o])


def _max_prob_btts(model: Probs, market: Probs) -> str:
    """BTTS's own belief arm, market ignored - _max_prob's twin."""
    return max(BTTS_OUTCOMES, key=lambda o: model[o])


#: rule name -> picks the side ("yes" or "no") to back on the BTTS market.
BTTS: dict[str, Callable[[Probs, Probs], str]] = {
    "max_edge": _max_edge_btts,
    "max_prob": _max_prob_btts,
}

#: Every rule, in the order reports should show them - the incumbent first so
#: a diff against it reads as "what changed", not "what is this".
RULES = ("max_edge", "max_prob")

assert set(RULES) == set(SINGLE) == set(DOUBLE_CHANCE) == set(BTTS), (
    "every rule needs a selector on every market - a rule present on only "
    "some would grade part of a fixture and silently skip the rest"
)
