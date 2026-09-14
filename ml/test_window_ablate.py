"""The sweep's arms differ only in which columns they select, so the column
composition is the whole experiment - a split arm silently taking its goals
from the wrong window would look like a finding."""
import window_ablate as wa
from feature_lib import FORM_FIELD_ORDER


def test_uniform_arm_takes_every_field_from_one_window():
    cols = wa.form_cols(12, 12)
    assert len(cols) == 2 * len(FORM_FIELD_ORDER)
    assert all(c.endswith("__w12") for c in cols)


def test_split_arm_takes_goals_long_and_stats_short():
    cols = wa.form_cols(38, 8)
    assert "home_form_goals_for__w38" in cols
    assert "away_form_goals_against__w38" in cols
    assert "home_form_possession__w8" in cols
    assert not any(c.startswith("home_form_possession__w38") for c in cols)
    # exactly four long columns: goals for/against, both sides
    assert sum(c.endswith("__w38") for c in cols) == 4


def test_every_arm_is_unique_and_complete():
    for name in wa.ARM_SPECS:
        cols = wa.arm_columns(name)
        assert len(cols) == len(set(cols)), name
        form = [c for c in cols if "_form_" in c]
        assert len(form) == 2 * len(FORM_FIELD_ORDER), name
        assert all(c in cols for c in wa.ATKDEF), name


def test_nocgr_arms_drop_only_club_goals_relative():
    full, stripped = set(wa.arm_columns("w8")), set(wa.arm_columns("w8_nocgr"))
    assert full - stripped == set(wa.CGR)


def test_baseline_arm_is_the_served_window():
    from feature_lib import MAX_WINDOW
    assert wa.ARM_SPECS[wa.BASELINE_ARM] == (MAX_WINDOW, MAX_WINDOW, True)
