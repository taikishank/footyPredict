import parse_lineups as pl


def _lineup_row(player_id: int, type_id: int = pl.STARTER_TYPE_ID, **extra) -> dict:
    row = {
        "fixture_id": 100,
        "team_id": 1,
        "player_id": player_id,
        "type_id": type_id,
        "position_id": 25,
        "player_name": f"Player {player_id}",
    }
    row.update(extra)
    return row


def _detail(type_id: int, value) -> dict:
    return {"type_id": type_id, "data": {"value": value}}


def test_lineup_source_none_when_no_lineup():
    assert pl.lineup_source({"lineups": []}) is None
    assert pl.lineup_source({}) is None


def test_lineup_source_predicted_without_bench():
    # a provider-predicted XI is 22 starters and no substitutes
    fixture = {"lineups": [_lineup_row(i) for i in range(22)]}
    assert pl.lineup_source(fixture) == pl.PREDICTED


def test_lineup_source_confirmed_once_bench_appears():
    fixture = {"lineups": [_lineup_row(1), _lineup_row(2, type_id=pl.BENCH_TYPE_ID)]}
    assert pl.lineup_source(fixture) == pl.CONFIRMED


def test_parse_lineups_marks_starters_and_bench():
    fixture = {"lineups": [
        _lineup_row(1, position_id=24, jersey_number=1, formation_position=1),
        _lineup_row(2, type_id=pl.BENCH_TYPE_ID),
        _lineup_row(3, type_id=999),  # neither starter nor bench - ignored
    ]}
    rows = pl.parse_lineups(fixture)
    assert [(r["player_id"], r["is_starting"]) for r in rows] == [(1, True), (2, False)]
    assert rows[0]["position"] == "goalkeeper"
    assert rows[0]["jersey_number"] == 1


def test_parse_player_stats_is_long_form():
    fixture = {"id": 100, "lineups": [
        _lineup_row(1, details=[_detail(119, 90), _detail(118, 7.5)]),
        _lineup_row(2, details=[_detail(119, 45)]),
    ]}
    rows = pl.parse_player_stats(fixture)
    assert len(rows) == 3  # one row per (player, type_id), not one per player
    assert {(r["player_id"], r["type_id"], r["value"]) for r in rows} == {
        (1, 119, 90), (1, 118, 7.5), (2, 119, 45),
    }


def test_parse_player_stats_absent_stat_yields_no_row():
    # absent means zero - the same contract extract_team_stats() uses. A player
    # who did not score simply has no Goals row; nothing is imputed here.
    fixture = {"id": 100, "lineups": [_lineup_row(1, details=[_detail(119, 90)])]}
    assert all(row["type_id"] != 52 for row in pl.parse_player_stats(fixture))


def test_parse_player_stats_keeps_non_numeric_values_separate():
    fixture = {"id": 100, "lineups": [
        _lineup_row(1, details=[_detail(40, True), _detail(119, 90)]),
    ]}
    by_type = {row["type_id"]: row for row in pl.parse_player_stats(fixture)}
    assert by_type[40]["value"] is None and by_type[40]["raw_value"] is True
    assert by_type[119]["value"] == 90 and by_type[119]["raw_value"] is None


def test_parse_players_falls_back_to_lineup_row_when_player_not_included():
    fixture = {"lineups": [_lineup_row(7, position_id=27)]}
    player, = pl.parse_players(fixture)
    assert player == {"player_id": 7, "name": "Player 7", "position": "attacker",
                      "date_of_birth": None, "raw": None}


def test_parse_players_prefers_the_included_player_object():
    fixture = {"lineups": [_lineup_row(7, player={
        "id": 7, "display_name": "Real Name", "position_id": 24, "date_of_birth": "1992-03-04",
    })]}
    player, = pl.parse_players(fixture)
    assert player["name"] == "Real Name"
    assert player["position"] == "goalkeeper"  # from the player object, not the lineup row
    assert player["date_of_birth"] == "1992-03-04"


def test_parse_players_deduplicates_within_a_fixture():
    fixture = {"lineups": [_lineup_row(7), _lineup_row(7, type_id=pl.BENCH_TYPE_ID)]}
    assert len(pl.parse_players(fixture)) == 1


def test_parse_sidelined_reads_the_nested_sideline_object():
    fixture = {"id": 100, "sidelined": [{
        "fixture_id": 100, "sideline_id": 792053, "player_id": 5,
        "participant_id": 20, "type_id": 537,
        "sideline": {"category": "injury", "team_id": 20, "start_date": "2026-05-15",
                     "end_date": "2026-06-01", "games_missed": 2},
    }]}
    row, = pl.parse_sidelined(fixture)
    assert row["team_id"] == 20          # participant_id, not team_id, at the top level
    assert row["category"] == "injury"
    assert row["reason_type_id"] == 537  # e.g. "Ankle Injury"
    assert row["games_missed"] == 2
    assert row["sideline_id"] == 792053


def test_parse_sidelined_survives_the_bare_include():
    # without the .sideline sub-include there are no dates - the row should
    # still be usable rather than blowing up
    fixture = {"id": 100, "sidelined": [
        {"fixture_id": 100, "player_id": 5, "participant_id": 20, "type_id": 537},
    ]}
    row, = pl.parse_sidelined(fixture)
    assert row["player_id"] == 5 and row["category"] is None and row["start_date"] is None


def test_team_totals_sums_per_team_and_type():
    fixture = {"id": 100, "lineups": [
        _lineup_row(1, team_id=1, details=[_detail(42, 3)]),
        _lineup_row(2, team_id=1, details=[_detail(42, 2)]),
        _lineup_row(3, team_id=2, details=[_detail(42, 4)]),
    ]}
    totals = pl.team_totals_from_players(fixture)
    assert totals[(1, 42)] == 5
    assert totals[(2, 42)] == 4
