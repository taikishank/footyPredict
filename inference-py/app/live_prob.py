"""Adjusts the pre-match model's probabilities using live score/state, since
the trained model (ml/train.py) only knows pre-match team form - it has no
notion of the current scoreline or clock. This is a hand-tuned heuristic
layer, not a learned one; see PROJECT_SPEC.md Phase 3 for the plan to
eventually replace it with a model trained on in-play features directly.

Heuristic: a goal's effect on the outcome logits grows as less time remains,
since a lead is harder to overturn late than early. Draw probability decays
with the same time weighting whenever the sides are level in the logit
space, since a growing lead crowds out the draw outcome.
"""
from __future__ import annotations

import math

# SportMonks fixture state short_names, mapped to a rough minutes-elapsed
# estimate. Coarse on purpose - see the module docstring - real minute data
# would need SportMonks' periods/events includes, which the live poller
# doesn't fetch yet.
_STATE_MINUTES_ELAPSED = {
    "NS": 0,
    "1H": 22,
    "HT": 45,
    "2H": 67,
    "FT": 90,
    "AWARDED": 90,
}
_DEFAULT_MINUTES_ELAPSED = 45  # unknown/unmapped state: assume mid-match
_REGULATION_MINUTES = 90

# Hand-tuned strengths of the adjustment; not fit to data.
# TODO: run manual tuning on these weights to improve model strength
# currently no training data to work with - 8/17/26
_SCORE_DIFF_WEIGHT = 1.5
_DRAW_DECAY_WEIGHT = 1.0

_PROB_CLIP = 1e-6


def adjust_for_live_state(
    prior: dict[str, float], state: str, home_goals: int, away_goals: int
) -> dict[str, float]:
    """Returns home_win/draw/away_win probabilities adjusted for the current
    scoreline and match state, given the pre-match model's `prior`."""
    minutes_elapsed = _STATE_MINUTES_ELAPSED.get(state, _DEFAULT_MINUTES_ELAPSED)
    minutes_remaining = max(0, _REGULATION_MINUTES - minutes_elapsed)
    time_weight = 1 - (minutes_remaining / _REGULATION_MINUTES)

    score_diff = home_goals - away_goals
    shift = _SCORE_DIFF_WEIGHT * score_diff * time_weight
    draw_decay = _DRAW_DECAY_WEIGHT * abs(score_diff) * time_weight

    logits = {label: _logit(p) for label, p in prior.items()}
    logits["home_win"] += shift
    logits["away_win"] -= shift
    logits["draw"] -= draw_decay

    return _softmax(logits)


def _logit(p: float) -> float:
    # Multinomial logit (plain log, not the binary log-odds form) - softmax
    # of these exactly reconstructs the input distribution when unshifted,
    # which is what makes a zero adjustment (level score) a no-op below.
    p = min(max(p, _PROB_CLIP), 1 - _PROB_CLIP)
    return math.log(p)


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    exps = {label: math.exp(x) for label, x in logits.items()}
    total = sum(exps.values())
    return {label: x / total for label, x in exps.items()}
