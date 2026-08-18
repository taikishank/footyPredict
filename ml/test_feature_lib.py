from feature_lib import FORM_FIELD_ORDER, MAX_WINDOW, STAT_IDS, extract_team_stats, team_form


def _stat_entry(type_id: int, participant_id: int, value: float) -> dict:
    return {"type_id": type_id, "participant_id": participant_id, "data": {"value": value}}


def test_extract_team_stats_ignores_other_participant():
    fixture = {
        "statistics": [
            _stat_entry(42, participant_id=1, value=10),  # shots_total for team 1
            _stat_entry(42, participant_id=2, value=99),  # shots_total for team 2 - ignored
        ]
    }
    stats = extract_team_stats(fixture, participant_id=1)
    assert stats["shots_total"] == 10


def test_extract_team_stats_derives_shot_pct_and_conversion_rate():
    fixture = {
        "statistics": [
            _stat_entry(42, 1, 10),  # shots_total
            _stat_entry(86, 1, 4),  # shots_on_target
            _stat_entry(580, 1, 3),  # big_chances_created
            _stat_entry(581, 1, 1),  # big_chances_missed
        ]
    }
    stats = extract_team_stats(fixture, participant_id=1)
    assert stats["shot_pct"] == 0.4
    assert stats["big_chances_conversion_rate"] == 0.75


def test_extract_team_stats_handles_zero_shots_without_dividing_by_zero():
    stats = extract_team_stats({"statistics": []}, participant_id=1)
    assert stats["shot_pct"] == 0.0
    assert stats["big_chances_conversion_rate"] == 0.0


def _history_entry(goals_for: int, goals_against: int) -> dict:
    return {
        "goals_for": goals_for,
        "goals_against": goals_against,
        "stats": {name: 0 for name in STAT_IDS.values()} | {
            "shot_pct": 0.0,
            "big_chances_conversion_rate": 0.0,
        },
    }


def test_team_form_averages_goals():
    history = [_history_entry(2, 1), _history_entry(0, 0), _history_entry(4, 2)]
    form = team_form(history)
    assert form["goals_for"] == 2.0
    assert form["goals_against"] == 1.0


def test_team_form_only_uses_max_window_most_recent_matches():
    history = [_history_entry(100, 0)] + [_history_entry(0, 0)] * MAX_WINDOW
    form = team_form(history)
    # the oldest (goals_for=100) entry should have been dropped by the window
    assert form["goals_for"] == 0.0


def test_form_field_order_matches_team_form_keys():
    history = [_history_entry(1, 1)] * 5
    form = team_form(history)
    assert list(form.keys()) == list(FORM_FIELD_ORDER)


def test_form_field_order_length_matches_trained_model_input_width():
    # ml/models/match_outcome_model.json expects 54 = 2 * len(FORM_FIELD_ORDER)
    # features (home_form_* + away_form_*). If this breaks, the model needs
    # retraining or FORM_FIELD_ORDER needs to change to match it.
    assert len(FORM_FIELD_ORDER) == 27
