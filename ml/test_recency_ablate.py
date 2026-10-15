"""The two things that would silently turn this phase into a false positive:
a weight computed from a date in the fold's own validation window (the
causal-ordering check this phase relies on instead of restating ra.folds'
own guarantee), and a placebo that does not actually preserve the weight
histogram it is supposed to control for."""
import numpy as np
import pandas as pd
import pytest

import recency_ablate as rec


def _frame(dates, league=None):
    return pd.DataFrame({
        "date": pd.to_datetime(dates),
        "league_id": league if league is not None else [8] * len(dates),
    })


def test_recency_weight_decays_with_age():
    df = _frame(["2020-01-01", "2020-06-01", "2020-12-01", "2021-06-01"])
    train_idx = np.array([0, 1, 2])
    val_idx = np.array([3])
    weight = rec.recency_weight(df, train_idx, val_idx, half_life=365.0)
    assert (np.diff(weight) > 0).all(), "a more recent row must weigh more"
    assert (weight > 0).all() and (weight <= 1.0).all()


def test_recency_weight_refuses_a_training_row_in_the_future():
    """The whole phase rests on never computing a weight from a date val_idx
    could not have known at serve time. A row dated after the fold's own
    asof date is exactly the leak ra.folds is supposed to make impossible."""
    df = _frame(["2020-01-01", "2021-01-01", "2020-06-01"])
    train_idx = np.array([0, 1])  # row 1 is AFTER val_idx's date
    val_idx = np.array([2])
    with pytest.raises(AssertionError):
        rec.recency_weight(df, train_idx, val_idx, half_life=365.0)


def test_recency_weight_is_relative_to_the_fold_not_a_global_today():
    """Two folds with different asof dates must weigh the same training row
    differently - otherwise this is secretly a single global decay, not a
    per-fold one, and later folds would leak how close to the present
    earlier folds are not supposed to know."""
    df = _frame(["2015-01-01", "2016-01-01", "2020-01-01", "2025-01-01"])
    early_weight = rec.recency_weight(df, np.array([0, 1]), np.array([2]), 365.0)
    late_weight = rec.recency_weight(df, np.array([0, 1]), np.array([3]), 365.0)
    assert (early_weight > late_weight).all()


def test_shuffled_weight_preserves_the_histogram_within_league():
    weight = np.array([1.0, 0.5, 0.25, 0.9, 0.1, 0.8])
    league = np.array([8, 8, 8, 82, 82, 82])
    shuffled = rec.shuffled_weight(weight, league, seed=0)
    for value in np.unique(league):
        rows = league == value
        assert sorted(shuffled[rows]) == sorted(weight[rows]), \
            "the within-league multiset must survive"
    assert not np.array_equal(shuffled, weight), \
        "a permutation that reproduces the identity defeats the placebo"


def test_shuffled_weight_does_not_leak_across_leagues():
    weight = np.array([1.0, 1.0, 0.0, 0.0])
    league = np.array([8, 8, 82, 82])
    shuffled = rec.shuffled_weight(weight, league, seed=1)
    assert set(shuffled[league == 8]) == {1.0}
    assert set(shuffled[league == 82]) == {0.0}


def test_anchor_weight_is_bounded_and_deterministic():
    df = _frame(["2015-01-01", "2016-01-01", "2020-01-01", "2024-01-01"])
    train_idx = np.array([0, 1, 2, 3])
    first = rec.anchor_weight(df, train_idx, half_life=365.0, seed=0)
    second = rec.anchor_weight(df, train_idx, half_life=365.0, seed=0)
    assert (first > 0).all() and (first <= 1.0).all()
    np.testing.assert_array_equal(first, second)


def test_effective_sample_size_matches_hand_computed_cases():
    uniform = np.ones(10)
    assert rec.effective_sample_size(uniform) == pytest.approx(10.0)

    one_dominant = np.array([1.0] + [1e-9] * 9)
    assert rec.effective_sample_size(one_dominant) == pytest.approx(1.0, abs=1e-3)

    half = np.array([1.0, 1.0, 0.5, 0.5])
    # (sum w)^2 / sum(w^2) by hand: (3.0)^2 / (1 + 1 + 0.25 + 0.25) = 9 / 2.5
    assert rec.effective_sample_size(half) == pytest.approx(9 / 2.5)


def test_fit_and_score_weighted_passes_sample_weight_through(monkeypatch):
    """The entire mechanism this phase tests is that sample_weight reaches
    .fit() and nothing else about the call changes - no column is added, the
    model class and params are the same shape every other arm uses."""
    df = pd.DataFrame({"x": range(6), "y_unused": range(6)})
    y = np.array([0, 1, 2, 0, 1, 2])
    train_idx = np.array([0, 1, 2])
    val_idx = np.array([3, 4, 5])
    weight = np.array([0.1, 0.5, 1.0])
    seen = {}

    class StubModel:
        def __init__(self, **kwargs):
            seen["init_kwargs"] = kwargs

        def fit(self, X, y, sample_weight=None):
            seen["sample_weight"] = sample_weight
            seen["fit_rows"] = len(X)

        def predict_proba(self, X):
            return np.full((len(X), 3), 1 / 3)

    monkeypatch.setattr(rec, "XGBClassifier", StubModel)
    params = {"max_depth": 2}
    rec.fit_and_score_weighted(df, ["x"], y, train_idx, val_idx, weight, params)

    np.testing.assert_array_equal(seen["sample_weight"], weight)
    assert seen["fit_rows"] == len(train_idx)
    assert seen["init_kwargs"]["max_depth"] == 2
