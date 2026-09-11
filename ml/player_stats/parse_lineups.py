"""Turn a SportMonks fixture payload into the row shapes Phase 8 stores.

Pure functions over already-fetched JSON: no network, no DB, so the shapes can
be unit-tested and re-derived from the cache without re-spending quota.

Player stats land in **long** form - one row per (fixture, player, type_id) -
rather than the wide, hand-picked columns `extract_team_stats()` uses for team
stats. The fixture payload carries ~60 stat types per player and we cache all
of them (see FINDINGS_DATA_AVAILABILITY.md), so a wide table would mean a
migration every time feature engineering wants one more. Long is one schema
forever; pivoting to a wide frame is a one-liner in pandas at feature-build
time.
"""

# /core/types: a lineup row is either a starter or a bench player. This is also
# the only confirmed-vs-predicted signal the payload offers - a predicted XI
# comes back as 22 STARTER rows with no BENCH rows at all.
STARTER_TYPE_ID = 11
BENCH_TYPE_ID = 12

# position_id values seen on lineup rows, mapped to the four groups
# squad_strength() (spec section 5) aggregates by.
POSITION_GROUPS = {
    24: "goalkeeper",
    25: "defender",
    26: "midfielder",
    27: "attacker",
    28: "unknown",  # "Unknown" in /core/types
}

CONFIRMED = "confirmed"
PREDICTED = "predicted"


def lineup_source(fixture: dict) -> str | None:
    """`confirmed`, `predicted`, or None when no lineup exists yet.

    Spec section 5 wants this flag carried through to the API response, since a
    prediction built off a provider-predicted XI is materially weaker than one
    built off a real team sheet. SportMonks exposes no explicit marker, so the
    bench is the discriminator: confirmed team sheets list substitutes,
    predicted XIs don't. Verified across 48 upcoming fixtures - see
    FINDINGS_DATA_AVAILABILITY.md Q3.
    """
    rows = fixture.get("lineups") or []
    if not any(row.get("type_id") == STARTER_TYPE_ID for row in rows):
        return None
    return CONFIRMED if any(row.get("type_id") == BENCH_TYPE_ID for row in rows) else PREDICTED


def parse_players(fixture: dict) -> list[dict]:
    """Player identity rows, deduplicated within the fixture.

    `lineups.player` gives the full player object; when only `lineups` was
    requested we still get player_id/player_name on the lineup row itself, so
    fall back to that rather than dropping the player entirely.
    """
    players: dict[int, dict] = {}
    for row in fixture.get("lineups") or []:
        player_id = row.get("player_id")
        if player_id is None:
            continue
        player = row.get("player") or {}
        players[player_id] = {
            "player_id": player_id,
            "name": player.get("display_name") or player.get("name") or row.get("player_name"),
            "position": POSITION_GROUPS.get(player.get("position_id") or row.get("position_id")),
            "date_of_birth": player.get("date_of_birth"),
            "raw": player or None,
        }
    return list(players.values())


def parse_lineups(fixture: dict) -> list[dict]:
    """One row per player named in the team sheet (starters and bench)."""
    rows = []
    for row in fixture.get("lineups") or []:
        type_id = row.get("type_id")
        if type_id not in (STARTER_TYPE_ID, BENCH_TYPE_ID):
            continue
        rows.append({
            "fixture_id": row.get("fixture_id") or fixture.get("id"),
            "team_id": row.get("team_id"),
            "player_id": row.get("player_id"),
            "is_starting": type_id == STARTER_TYPE_ID,
            "position": POSITION_GROUPS.get(row.get("position_id")),
            "position_id": row.get("position_id"),
            "formation_field": row.get("formation_field"),
            "formation_position": row.get("formation_position"),
            "jersey_number": row.get("jersey_number"),
        })
    return rows


def parse_player_stats(fixture: dict) -> list[dict]:
    """Long-form player stats: one row per (fixture, player, type_id).

    Absent means zero here, same contract as `extract_team_stats()` - a team
    that scored no goals produces no Goals rows at all, so a missing row must
    not be read as missing data. Nothing is imputed at this layer; the feature
    builder fills the zeros when it pivots.
    """
    fixture_id = fixture.get("id")
    rows = []
    for lineup_row in fixture.get("lineups") or []:
        player_id = lineup_row.get("player_id")
        for detail in lineup_row.get("details") or []:
            value = (detail.get("data") or {}).get("value")
            rows.append({
                "fixture_id": lineup_row.get("fixture_id") or fixture_id,
                "player_id": player_id,
                "team_id": lineup_row.get("team_id"),
                "type_id": detail.get("type_id"),
                # SportMonks sends ints, floats (ratings, percentages) and the
                # occasional bool/None through the same field, so keep a
                # numeric column for the common case and the raw value for the
                # rest rather than silently coercing.
                "value": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                "raw_value": None if isinstance(value, (int, float)) and not isinstance(value, bool) else value,
            })
    return rows


def parse_sidelined(fixture: dict) -> list[dict]:
    """Injury/suspension rows, as of this fixture.

    These are scoped to the fixture rather than "current", which is what makes
    them backfillable: a standalone current-state table could not be
    reconstructed historically without leaking information that wasn't known
    at kickoff. /squads and a standalone /sidelined endpoint are both
    unavailable on our plan anyway (Q1).
    """
    rows = []
    for entry in fixture.get("sidelined") or []:
        # the dates/category live on the nested `sideline` object, which only
        # arrives with the `sidelined.sideline` sub-include - the bare
        # `sidelined` include gives ids and nothing else. Note the team is
        # `participant_id` here, not `team_id` as elsewhere in the payload.
        detail = entry.get("sideline") or {}
        rows.append({
            "fixture_id": entry.get("fixture_id") or fixture.get("id"),
            # a player can carry more than one sideline record into the same
            # fixture, so the sideline id is what makes a row uniquely keyed
            "sideline_id": entry.get("sideline_id") or detail.get("id"),
            "player_id": entry.get("player_id") or detail.get("player_id"),
            "team_id": entry.get("participant_id") or detail.get("team_id"),
            "category": detail.get("category"),          # injury / suspension / ...
            "reason_type_id": entry.get("type_id"),      # e.g. 537 "Ankle Injury"
            "start_date": detail.get("start_date"),
            "end_date": detail.get("end_date"),
            "games_missed": detail.get("games_missed"),
            "raw": entry,
        })
    return rows


def team_totals_from_players(fixture: dict) -> dict[tuple[int, int], float]:
    """Sum player stats per (team_id, type_id).

    A cross-check, not a replacement for the team `statistics` include: most
    counting stats reconcile exactly, but 11 of the 23 STAT_IDS the current
    model uses have no player-level equivalent at all (possession, corners,
    red cards, ...) and Shots Off Target is defined differently at the two
    grains. See FINDINGS_DATA_AVAILABILITY.md. Useful for spotting a fixture
    whose lineup payload came back incomplete.
    """
    totals: dict[tuple[int, int], float] = {}
    for row in parse_player_stats(fixture):
        if row["value"] is None or row["team_id"] is None:
            continue
        key = (row["team_id"], row["type_id"])
        totals[key] = totals.get(key, 0) + row["value"]
    return totals
