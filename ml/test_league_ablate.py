"""The arms differ only in which columns they select, and per_league differs
in how it fits. Both are places where a silent bug reads as a finding: an arm
that leaked league_id into the baseline would make every other arm look
worthless, and a per_league arm that returned its losses in league order
rather than fixture order would be bootstrapped against the wrong rows."""
import numpy as np
import pandas as pd
import pytest

import league_ablate as la
from league_lib import LeagueCharacter

BASE = ["home_form_goals_for", "exp_goal_diff"]
DUMMIES = ["league_is_8", "league_is_82"]
PLACEBO = ["fake_is_0", "fake_is_1"]


def test_baseline_carries_no_league_column():
    cols = la.arm_columns(la.BASELINE, BASE, DUMMIES)
    assert cols == BASE
    assert not any("league" in c for c in cols)


def test_per_league_gets_the_baseline_columns_exactly():
    """Its league information is the fit, not a column. Handing it a dummy as
    well would be a constant within each model and pure noise."""
    assert la.arm_columns(la.PER_LEAGUE, BASE, DUMMIES, PLACEBO) == BASE


def test_interact_is_a_strict_superset_of_onehot():
    onehot = set(la.arm_columns("onehot", BASE, DUMMIES))
    interact = set(la.arm_columns("interact", BASE, DUMMIES))
    assert onehot < interact
    assert interact - onehot == {
        f"{d}_x_{s}" for d in DUMMIES for s in la.INTERACT_WITH
    }


def test_character_arm_adds_the_four_rolling_columns_and_no_identity():
    cols = la.arm_columns("character", BASE, DUMMIES)
    assert set(cols) - set(BASE) == set(LeagueCharacter.FIELD_ORDER)
    assert "league_id" not in cols
    assert not any(c.startswith("league_is_") for c in cols)


def test_placebo_has_interacts_shape_and_none_of_its_content():
    """The control is only a control if it costs the model the same number of
    columns; a narrower placebo would fail for the wrong reason."""
    interact = la.arm_columns("interact", BASE, DUMMIES, PLACEBO)
    placebo = la.arm_columns("placebo", BASE, DUMMIES, PLACEBO)
    assert len(placebo) == len(interact)
    assert not any("league" in c for c in set(placebo) - set(BASE))


def test_every_arm_is_unique_and_keeps_the_baseline():
    for name in la.ARMS_ORDER:
        cols = la.arm_columns(name, BASE, DUMMIES, PLACEBO)
        assert len(cols) == len(set(cols)), name
        assert all(c in cols for c in BASE), name


def test_per_league_losses_come_back_in_validation_order(monkeypatch):
    """Each league is fitted in its own pass, so the scatter back into
    val_idx's order is the only thing keeping this arm paired with the rest.
    The stub returns the row's own index, so the returned array must read
    exactly as the validation indices themselves."""
    df = pd.DataFrame({
        "league_id": [8, 82, 8, 82, 8, 82, 8, 82],
        "x": range(8),
    })
    y = np.zeros(8, dtype=int)
    train_idx, val_idx = np.arange(4), np.array([5, 4, 7, 6])

    def stub(frame, cols, labels, train, val):
        assert set(frame["league_id"].to_numpy()[train]) == \
            {frame["league_id"].to_numpy()[val][0]}, "trained on another league"
        return val.astype(float)

    monkeypatch.setattr(la.ra, "fit_and_score", stub)
    losses = la.fit_and_score_per_league(df, ["x"], y, train_idx, val_idx)
    assert losses.tolist() == val_idx.tolist()


def test_per_league_refuses_a_league_it_never_trained_on():
    df = pd.DataFrame({"league_id": [8, 8, 301, 301], "x": range(4)})
    y = np.zeros(4, dtype=int)
    with pytest.raises(ValueError, match="no training rows"):
        la.fit_and_score_per_league(df, ["x"], y, np.array([0, 1]), np.array([2, 3]))
