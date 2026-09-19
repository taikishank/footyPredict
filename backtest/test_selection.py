"""Properties of the selection rules that the ablation depends on being true.

Run: ./inference-py/bin/python -m pytest backtest/test_selection.py
"""
from __future__ import annotations

import pytest

import selection

OUT = selection.OUTCOMES


def _market(h: float, d: float, a: float) -> dict[str, float]:
    """A de-vigged market, i.e. one that sums to 1 - which is what
    ingestor-go/internal/odds/parse.go writes and what makes the edges sum
    to 0. Normalized here so a test cannot accidentally assert the
    structural property against a market that does not have it."""
    total = h + d + a
    return dict(zip(OUT, (h / total, d / total, a / total)))


def _model(h: float, d: float, a: float) -> dict[str, float]:
    total = h + d + a
    return dict(zip(OUT, (h / total, d / total, a / total)))


# The case the whole exercise is about: v3 prices the draw above the market
# on 96.8% of snapshots, so max_edge backs a draw the model does not rank
# first, while max_prob backs the home win it does.
DRAW_BIAS_MODEL = _model(0.45, 0.31, 0.24)
DRAW_BIAS_MARKET = _market(0.50, 0.25, 0.25)


def test_max_edge_picks_a_draw_the_model_does_not_favour():
    pick = selection.SINGLE["max_edge"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    assert pick == "draw"
    # Not that the draw is ranked last - that it is not ranked first.
    # Measured on v3: 387 of 777 picks were draws, and v3 ranks the draw
    # first on 0 of 777.
    assert DRAW_BIAS_MODEL[pick] < max(DRAW_BIAS_MODEL.values())


def test_max_prob_picks_the_model_favourite():
    pick = selection.SINGLE["max_prob"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    assert pick == "home_win"
    assert DRAW_BIAS_MODEL[pick] == max(DRAW_BIAS_MODEL.values())


def test_the_two_rules_disagree_here():
    assert selection.SINGLE["max_edge"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET) != \
        selection.SINGLE["max_prob"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)


@pytest.mark.parametrize(
    "model",
    [
        _model(0.45, 0.31, 0.24),
        _model(0.80, 0.12, 0.08),
        _model(0.10, 0.20, 0.70),
        _model(1 / 3, 1 / 3, 1 / 3),
    ],
)
@pytest.mark.parametrize(
    "market",
    [_market(0.50, 0.25, 0.25), _market(0.20, 0.25, 0.55), _market(0.34, 0.33, 0.33)],
)
def test_max_edge_can_never_produce_a_negative_edge(model, market):
    """The structural claim the ablation exists to expose: against a de-vigged
    market the three edges sum to 0, so max_edge's pick is never negative.
    "The model found an edge here" is therefore not information."""
    pick = selection.SINGLE["max_edge"](model, market)
    assert model[pick] - market[pick] >= -1e-12
    assert sum(model[o] - market[o] for o in OUT) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("rule", selection.RULES)
def test_every_rule_always_names_a_side(rule):
    """No rule shipped so far refuses, which is what keeps the arms comparable
    on one sample rather than on differently-sized slices."""
    model, market = DRAW_BIAS_MODEL, DRAW_BIAS_MARKET
    assert selection.SINGLE[rule](model, market) in OUT
    assert selection.DOUBLE_CHANCE[rule](model, market) in OUT


def test_dc_max_prob_excludes_the_least_likely_outcome():
    excluded = selection.DOUBLE_CHANCE["max_prob"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    assert excluded == "away_win"
    assert DRAW_BIAS_MODEL[excluded] == min(DRAW_BIAS_MODEL.values())


def test_dc_max_edge_excludes_the_most_overpriced_outcome():
    excluded = selection.DOUBLE_CHANCE["max_edge"](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    edges = {o: DRAW_BIAS_MODEL[o] - DRAW_BIAS_MARKET[o] for o in OUT}
    assert edges[excluded] == min(edges.values())


@pytest.mark.parametrize("rule", selection.RULES)
def test_dc_is_not_derived_from_the_singles_pick(rule):
    """The DC arm must be able to cover the singles pick, not merely negate
    it - when it was "the pair excluding call_side", dc_correct was the exact
    negation of correct and the arm measured nothing new."""
    single = selection.SINGLE[rule](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    excluded = selection.DOUBLE_CHANCE[rule](DRAW_BIAS_MODEL, DRAW_BIAS_MARKET)
    assert single != excluded, "DC excluded the very outcome the singles arm backed"
