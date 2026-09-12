"""Stage 2: player-match rows -> one feature row per fixture, joined to the
incumbent's team-form table.

The output is deliberately a superset of `data/features/2000_window3_8.csv`:
same rows, same team-form columns, plus the availability and performance blocks
from `player_feature_lib`. That is what makes the Phase 9 comparison honest -
the challenger and the rebaselined incumbent are then trained and validated on
exactly the same fixtures, so any difference is the new features rather than a
different slice of history (spec section 6's promotion bar).

Two output tables, one per tier:

  challenger_availability.csv  team form + availability block, 2006+
  challenger_performance.csv   team form + both blocks, 2016+

Leakage is prevented structurally, not by inspection: state is updated only
*after* a fixture's row has been emitted, exactly as `build_features.py`
appends to `team_history` after using it.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from player_feature_lib import (  # noqa: E402
    AVAILABILITY_FIELD_ORDER,
    MIN_APPEARANCES,
    PERFORMANCE_FIELD_ORDER,
    SQUAD_RECENCY_WINDOW,
    ClubInvolvements,
    LeagueRollingMean,
    PositionAverages,
    absence_impact,
    availability_form,
    impute_form,
    involvement_form,
    key_players,
    numeric,
    player_form,
    squad_strength,
)

FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
PLAYER_TABLE = FEATURES_DIR / "player_matches.parquet"
TEAM_FEATURES = FEATURES_DIR / "2000_window3_8.csv"

AVAILABILITY_OUT = FEATURES_DIR / "challenger_availability.csv"
PERFORMANCE_OUT = FEATURES_DIR / "challenger_performance.csv"

# Minutes played is populated from 2006 and ratings from 2016; see
# FINDINGS_DATA_AVAILABILITY.md. Cutting at the season boundary rather than the
# calendar year keeps a season from being split across the two tiers.
AVAILABILITY_FROM = "2006-07-01"
PERFORMANCE_FROM = "2016-07-01"

# Columns from the player table that player_form() consumes.
_STAT_COLUMNS = None  # resolved from the parquet at load time


def _fixture_sides(df: pd.DataFrame):
    """Yield (fixture_id, date, league_id, {team_id: rows}) in date order.

    Rows come back as plain dicts because everything downstream is pure-Python
    set and mean arithmetic over ~22 players; staying in pandas for that would
    be slower, not faster.
    """
    columns = list(df.columns)
    for (fixture_id, date, league_id), group in df.groupby(
        ["fixture_id", "date", "league_id"], sort=False
    ):
        teams: dict[int, list[dict]] = {}
        for values in group.itertuples(index=False, name=None):
            row = dict(zip(columns, values))
            teams.setdefault(row["team_id"], []).append(row)
        yield fixture_id, date, league_id, teams


def _recent_squad(squads: list[frozenset[int]]) -> set[int]:
    """Everyone who has played for the club across its recent team sheets."""
    return {player_id for squad in squads for player_id in squad}


def _involvements(team_rows: list[dict]) -> dict[int, float]:
    """Goals + assists per player in one match, omitting the uninvolved."""
    counted = {}
    for row in team_rows:
        total = (numeric(row.get("goals")) or 0.0) + (numeric(row.get("assists")) or 0.0)
        if total > 0:
            counted[row["player_id"]] = total
    return counted


def build() -> pd.DataFrame:
    df = pd.read_parquet(PLAYER_TABLE).sort_values(["date", "fixture_id"])

    player_history: dict[int, list[dict]] = {}
    appearances: dict[int, int] = {}
    recent_xis: dict[int, list[frozenset[int]]] = {}
    season_minutes: dict[tuple[int, int], dict[int, float]] = {}
    last_played: dict[int, pd.Timestamp] = {}
    recent_squads: dict[int, list[frozenset[int]]] = {}
    averages = PositionAverages()
    league_experience = LeagueRollingMean()
    involvements = ClubInvolvements()
    # One reference per volume column. Both are sums over a 38-match window, so
    # both inherit the archive's density ramp and both need dividing out.
    league_volume = LeagueRollingMean()
    league_goals = LeagueRollingMean()

    rows = []
    for fixture_id, date, league_id, teams in _fixture_sides(df):
        # Fixtures where one side's team sheet never arrived cannot produce a
        # both-sides feature row; skipping keeps the caller from having to
        # reason about half-populated vectors.
        if len(teams) != 2:
            continue

        season_id = next(iter(teams.values()))[0]["season_id"]
        blocks: dict[int, dict] = {}
        for team_id, team_rows in teams.items():
            xi = [r for r in team_rows if r["is_starting"]]
            if not xi:
                continue

            gap = last_played.get(team_id)
            available = availability_form(
                xi=xi,
                recent_xis=recent_xis.get(team_id, []),
                season_minutes=season_minutes.get((team_id, season_id), {}),
                appearances=appearances,
                days_since_last_match=(date - gap).days if gap is not None else None,
                league_reference=league_experience.reference(league_id),
            )

            # Forms are needed for the absent key players too, not just the
            # eleven on the pitch: an absence feature weighted by the missing
            # player's own rating has to look him up somewhere.
            team_minutes = season_minutes.get((team_id, season_id), {})
            xi_ids = {r["player_id"] for r in xi}

            shares, _sample = involvements.shares(team_id)
            available.update(
                involvement_form(
                    xi=xi,
                    key_ids=key_players(team_minutes),
                    recent_squad=_recent_squad(recent_squads.get(team_id, [])),
                    shares=shares,
                    volumes=involvements.volumes(team_id),
                    references=(
                        league_volume.reference(league_id),
                        league_goals.reference(league_id),
                    ),
                )
            )
            absent = [
                {"player_id": pid, "position": player_history[pid][-1]["position"]}
                for pid in key_players(team_minutes)
                if pid not in xi_ids and player_history.get(pid)
            ]
            forms = {
                pid: player_form(player_history[pid])
                for pid in xi_ids | {row["player_id"] for row in absent}
                if len(player_history.get(pid, ())) >= MIN_APPEARANCES
            }
            fallback = impute_form(averages, league_id)
            strength = squad_strength(xi, forms, fallback, weights=team_minutes)
            missing = absence_impact(absent, team_minutes, forms, fallback)

            blocks[team_id] = {**available, **strength, **missing}

        if len(blocks) == 2:
            rows.append((fixture_id, date, blocks))

        # --- state update: everything below is strictly after the emit above
        for team_id, team_rows in teams.items():
            starting_ids = {r["player_id"] for r in team_rows if r["is_starting"]}
            if starting_ids:
                league_experience.update(
                    league_id,
                    sum(appearances.get(p, 0) for p in starting_ids) / len(starting_ids),
                )
            starters = frozenset(r["player_id"] for r in team_rows if r["is_starting"])
            recent_xis.setdefault(team_id, []).append(starters)

            # Substitutes included: a club's goalscorers are not all starters,
            # and "is he still at the club" is a question about the squad.
            played_ids = frozenset(
                r["player_id"] for r in team_rows if (numeric(r.get("minutes_played")) or 0) > 0
            )
            squads = recent_squads.setdefault(team_id, [])
            squads.append(played_ids)
            if len(squads) > SQUAD_RECENCY_WINDOW:
                del squads[:-SQUAD_RECENCY_WINDOW]

            scored = _involvements(team_rows)
            goals = sum(numeric(r.get("goals")) or 0.0 for r in team_rows)
            involvements.update(team_id, scored, goals)

            # The league references see this club's *updated* window, which is
            # correct: they are a level to compare against, not a prediction,
            # and every fixture's own features were emitted above before any of
            # this ran.
            volume, goal_volume = involvements.volumes(team_id)
            league_volume.update(league_id, volume)
            league_goals.update(league_id, goal_volume)
            minutes = season_minutes.setdefault((team_id, season_id), {})
            for r in team_rows:
                played = numeric(r.get("minutes_played"))
                # No minutes recorded is not zero minutes: a bench player who
                # never came on and one whose minutes the provider never
                # reported are indistinguishable here, and counting either as
                # an appearance would dilute every form window with matches the
                # player may not have played.
                if not played or played <= 0:
                    continue
                minutes[r["player_id"]] = minutes.get(r["player_id"], 0.0) + played
                history = player_history.setdefault(r["player_id"], [])
                history.append(r)
                appearances[r["player_id"]] = appearances.get(r["player_id"], 0) + 1
                averages.update(league_id, r["position"], player_form(history))
            last_played[team_id] = date

    return rows


def to_frame(rows: list, sides: pd.DataFrame, team_features: pd.DataFrame) -> pd.DataFrame:
    """Attach each fixture's two team blocks to the incumbent's row as home/away.

    Joining onto the incumbent's table is not just convenience: it guarantees
    the two models see the same fixtures. Rows the incumbent never produced -
    a club still below MIN_WINDOW, typically newly promoted - are dropped here
    for the same reason, since the challenger cannot be credited with fixtures
    the baseline never got to predict.
    """
    side_of = {
        r.fixture_id: {r.home_id: "home", r.away_id: "away"}
        for r in sides.itertuples(index=False)
    }

    flat = []
    for fixture_id, _date, blocks in rows:
        locations = side_of.get(fixture_id)
        if locations is None:
            continue
        record = {"fixture_id": fixture_id}
        for team_id, block in blocks.items():
            location = locations.get(team_id)
            if location is None:
                break
            for name, value in block.items():
                record[f"{location}_sq_{name}"] = value
        else:
            flat.append(record)

    player_cols = pd.DataFrame(flat)
    return team_features.merge(player_cols, on="fixture_id", how="inner")


AVAILABILITY_COLUMNS = tuple(
    f"{side}_sq_{name}" for side in ("home", "away") for name in AVAILABILITY_FIELD_ORDER
)
PERFORMANCE_COLUMNS = tuple(
    f"{side}_sq_{name}" for side in ("home", "away") for name in PERFORMANCE_FIELD_ORDER
)


def main() -> None:
    rows = build()
    sides = pd.read_parquet(FEATURES_DIR / "fixture_sides.parquet")
    team_features = pd.read_csv(TEAM_FEATURES, parse_dates=["date"])
    df = to_frame(rows, sides, team_features)
    print(f"{len(df):,} fixtures with both a team-form row and two team sheets")

    base = [c for c in team_features.columns]

    available = df[df["date"] >= AVAILABILITY_FROM][base + list(AVAILABILITY_COLUMNS)]
    available.to_csv(AVAILABILITY_OUT, index=False)
    print(f"wrote {len(available):,} rows x {available.shape[1]} cols to {AVAILABILITY_OUT}")

    performance = df[df["date"] >= PERFORMANCE_FROM][
        base + list(AVAILABILITY_COLUMNS) + list(PERFORMANCE_COLUMNS)
    ]
    performance.to_csv(PERFORMANCE_OUT, index=False)
    print(f"wrote {len(performance):,} rows x {performance.shape[1]} cols to {PERFORMANCE_OUT}")


if __name__ == "__main__":
    main()
