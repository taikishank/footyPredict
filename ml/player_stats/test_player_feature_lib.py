import math

import player_feature_lib as pf


def _appearance(minutes=90, rating=7.0, **stats) -> dict:
    row = {"minutes_played": minutes, "rating": rating}
    row.update(dict.fromkeys(pf.PER90_STATS, 0))
    row.update(stats)
    return row


def _starter(player_id: int, position: str = "midfielder") -> dict:
    return {"player_id": player_id, "position": position}


# --- numeric coercion --------------------------------------------------------


def test_numeric_treats_nan_as_missing():
    # the bug this guards: NaN is truthy, so `nan or 0` is nan, not 0, and one
    # unrecorded minute count turns an entire feature block into NaN.
    assert pf.numeric(float("nan")) is None
    assert pf.numeric(None) is None
    assert pf.numeric(True) is None
    assert pf.numeric("7.2") is None
    assert pf.numeric(0) == 0.0
    assert pf.numeric(7) == 7.0


# --- player_form -------------------------------------------------------------


def test_player_form_rates_are_per_90_not_per_match():
    # two goals in one full match and none in the next is 1.0 per 90, not 2.0
    history = [_appearance(goals=2), _appearance(goals=0)]
    assert pf.player_form(history)["goals"] == 1.0


def test_player_form_does_not_extrapolate_a_cameo():
    # a goal in five minutes must not read as eighteen goals per 90 - the whole
    # window's minutes are the denominator, not the one match's
    history = [_appearance(minutes=90, goals=0)] * 9 + [_appearance(minutes=5, goals=1)]
    assert pf.player_form(history)["goals"] < 0.2


def test_player_form_honours_the_appearance_window():
    history = [_appearance(rating=6.0)] * 20 + [_appearance(rating=8.0)] * 10
    assert pf.player_form(history)["rating"] == 8.0


def test_player_form_skips_missing_ratings_rather_than_zeroing_them():
    history = [_appearance(rating=None), _appearance(rating=8.0)]
    assert pf.player_form(history)["rating"] == 8.0


def test_player_form_of_an_unseen_player_is_all_none():
    assert all(v is None for v in pf.player_form([]).values())


def test_player_form_survives_unrecorded_minutes():
    form = pf.player_form([_appearance(minutes=float("nan"), goals=1)])
    assert form["goals"] == 0.0
    assert not math.isnan(form["minutes_per_appearance"])


# --- modal XI ----------------------------------------------------------------


def test_modal_xi_ignores_a_single_rotated_lineup():
    first_choice = frozenset(range(11))
    rotated = frozenset(range(100, 111))
    assert pf.modal_xi([first_choice] * 4 + [rotated]) == set(first_choice)


def test_modal_xi_breaks_ties_towards_the_recent_side():
    old = frozenset(range(11))
    new = frozenset(range(100, 111))
    # both started twice; the newer set should own the eleven places
    assert pf.modal_xi([old, old, new, new]) == set(new)


# --- availability ------------------------------------------------------------


def test_availability_measures_retention_against_the_settled_xi():
    settled = [frozenset(range(11))] * 5
    xi = [_starter(i) for i in range(11)]
    block = pf.availability_form(xi, settled, {}, {}, None, None)
    assert block["modal_xi_retained"] == 1.0
    assert block["prev_xi_retained"] == 1.0

    changed = [_starter(i) for i in range(6)] + [_starter(i) for i in range(100, 105)]
    block = pf.availability_form(changed, settled, {}, {}, None, None)
    assert block["modal_xi_retained"] == 6 / 11


def test_availability_counts_absent_key_players():
    # players 0-10 are the season's most-used; this XI is missing three of them
    season_minutes = {pid: 1000 - pid for pid in range(15)}
    xi = [_starter(i) for i in range(8)] + [_starter(i) for i in range(50, 53)]
    block = pf.availability_form(xi, [], season_minutes, {}, None, None)
    assert block["key_players_absent"] == 3


def test_availability_reports_no_absences_before_a_season_has_minutes():
    # matchday one has no minutes to rank by; eleven absences would be wrong
    xi = [_starter(i) for i in range(11)]
    assert pf.availability_form(xi, [], {}, {}, None, None)["key_players_absent"] == 0


def test_availability_counts_debutants():
    appearances = {0: 50, 1: 50, 2: 1}
    xi = [_starter(0), _starter(1), _starter(2), _starter(3)]
    assert pf.availability_form(xi, [], {}, appearances, None, None)["xi_debutants"] == 2


def test_availability_experience_is_relative_to_the_league():
    xi = [_starter(i) for i in range(11)]
    appearances = dict.fromkeys(range(11), 120)
    block = pf.availability_form(xi, [], {}, appearances, None, league_reference=60.0)
    assert block["xi_relative_experience"] == 2.0
    # with no reference yet the feature is absent rather than a raw count that
    # would smuggle the observation window's warm-up in as a date proxy
    assert pf.availability_form(xi, [], {}, appearances, None, None)[
        "xi_relative_experience"
    ] is None


def test_availability_emits_every_declared_field():
    xi = [_starter(i) for i in range(11)]
    block = pf.availability_form(xi, [], {}, {}, 7, 100.0)
    assert set(block) == set(pf.AVAILABILITY_FIELD_ORDER)


# --- squad strength ----------------------------------------------------------


def _no_history(row):
    return dict.fromkeys(pf.PLAYER_FORM_FIELDS, None)


def test_squad_strength_emits_every_declared_field():
    xi = [_starter(i) for i in range(11)]
    block = pf.squad_strength(xi, {}, _no_history)
    assert set(block) == set(pf.PERFORMANCE_FIELD_ORDER)


def test_squad_strength_counts_positions_as_the_shape_of_the_xi():
    xi = (
        [_starter(0, "goalkeeper")]
        + [_starter(i, "defender") for i in range(1, 4)]
        + [_starter(i, "midfielder") for i in range(4, 9)]
        + [_starter(i, "attacker") for i in range(9, 11)]
    )
    block = pf.squad_strength(xi, {}, _no_history)
    assert (block["defender_count"], block["midfielder_count"]) == (3, 5)


def test_squad_strength_falls_back_and_says_how_often():
    xi = [_starter(i) for i in range(11)]
    forms = {i: pf.player_form([_appearance(rating=7.5)] * 5) for i in range(9)}
    average = {**dict.fromkeys(pf.PLAYER_FORM_FIELDS, 0.0), "rating": 6.0}
    block = pf.squad_strength(xi, forms, lambda row: average)
    assert block["imputed_players"] == 2
    assert block["midfielder_rating"] == (9 * 7.5 + 2 * 6.0) / 11


def test_squad_strength_sums_output_so_a_swap_moves_the_number():
    scorer = pf.player_form([_appearance(goals=1)])
    xi = [_starter(i) for i in range(11)]
    forms = dict.fromkeys(range(11), pf.player_form([_appearance(goals=0)]))
    lean = pf.squad_strength(xi, forms, _no_history)["attack_goals"]
    forms[0] = scorer
    assert pf.squad_strength(xi, forms, _no_history)["attack_goals"] > lean


def test_rating_spread_needs_two_ratings():
    xi = [_starter(0)]
    forms = {0: pf.player_form([_appearance(rating=7.0)])}
    assert pf.squad_strength(xi, forms, _no_history)["rating_spread"] is None


# --- imputation sources ------------------------------------------------------


def test_position_averages_are_as_of_and_position_scoped():
    averages = pf.PositionAverages()
    assert averages.get(8, "defender")["rating"] is None  # nothing seen yet

    averages.update(8, "defender", pf.player_form([_appearance(rating=6.0)]))
    averages.update(8, "defender", pf.player_form([_appearance(rating=8.0)]))
    averages.update(8, "attacker", pf.player_form([_appearance(rating=5.0)]))
    assert averages.get(8, "defender")["rating"] == 7.0
    assert averages.get(8, "attacker")["rating"] == 5.0


def test_position_averages_fall_back_across_an_unseen_position():
    averages = pf.PositionAverages()
    averages.update(8, None, pf.player_form([_appearance(rating=6.5)]))
    # a player whose position the payload never recorded still gets a number
    assert averages.get(8, "attacker")["rating"] == 6.5


def test_position_averages_ignore_players_with_no_rating():
    averages = pf.PositionAverages()
    averages.update(8, "defender", pf.player_form([]))
    assert averages.get(8, "defender")["rating"] is None


def test_league_experience_tracks_the_ramp_rather_than_lagging_it():
    experience = pf.LeagueExperience()
    assert experience.reference(8) is None
    for _ in range(pf.LeagueExperience.WINDOW * 2):
        experience.update(8, 20.0)
    assert experience.reference(8) == 20.0
    for _ in range(pf.LeagueExperience.WINDOW):
        experience.update(8, 120.0)
    # the early, low-appearance era has aged out entirely
    assert experience.reference(8) == 120.0
