"""Stage 1 of the Phase 9 pipeline: the cached lineup payloads, flattened.

`ml/data/raw/player_lineups/*.json.gz` holds 41,777 fixtures of team sheets and
~60 stat types per player (Phase 8). Parsing all of it takes ~90 seconds, which
is far too slow to sit inside the feature-design loop, so this runs once and
writes a single parquet that `build_player_features.py` reads in a couple of
seconds. Re-run it only when the cache itself changes.

One row per (fixture, player) in the team sheet, wide over the stat types
listed in STAT_TYPES. Wide rather than the long form `parse_lineups.py`
produces, because from here on every consumer wants "this player's rating in
this match" as a column; the long form's advantage - never migrating when a new
stat type matters - is preserved by this script being cheap to re-run against
the untouched cache.
"""

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from parse_lineups import (  # noqa: E402
    BENCH_TYPE_ID,
    POSITION_GROUPS,
    STARTER_TYPE_ID,
    lineup_source,
)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "player_lineups"
FEATURES_DIR = Path(__file__).parent.parent / "data" / "features"
OUT_PATH = FEATURES_DIR / "player_matches.parquet"

# Which team was at home is not in the lineup payload at all, only in the
# fixture cache the incumbent reads. Stage 2 needs it to turn two anonymous
# team blocks into home_/away_ columns, so it is derived here, in the one
# stage that is allowed to be slow.
ALL_FIXTURES_DIR = Path(__file__).parent.parent / "data" / "raw" / "all_fixtures"
SIDES_PATH = FEATURES_DIR / "fixture_sides.parquet"

# SportMonks type_id -> column name. The cache holds ~60 types; these are the
# ones the two feature tiers need, plus a few cheap extras that cost one column
# each here and would cost a full re-parse to add later.
#
# Tier split (see the 2016 stats cliff in FINDINGS_DATA_AVAILABILITY.md):
#   minutes_played is populated from 2006 onward and drives the availability
#   tier; everything else below is effectively 2016+ only and drives the
#   performance tier.
STAT_TYPES = {
    119: "minutes_played",
    118: "rating",
    52: "goals",
    79: "assists",
    42: "shots_total",
    86: "shots_on_target",
    117: "key_passes",
    78: "tackles",
    100: "interceptions",
    101: "clearances",
    106: "duels_won",
    109: "successful_dribbles",
    9706: "chances_created",
    580: "big_chances_created",
    88: "goals_conceded",
    57: "saves",
    84: "yellow_cards",
    83: "red_cards",
}

# Absent means zero, the same contract extract_team_stats() uses for team
# stats - a player who took no shots produces no Shots Total row at all. The
# exceptions are the two that are genuinely unmeasured rather than zero:
# a rating of 0 would be a wrong claim about how the player performed, and
# 0 minutes would make a starter look unused. Both stay null so the feature
# layer can skip them rather than average a fiction in.
NOT_ZERO_FILLED = ("rating", "minutes_played")


def flatten_fixture(fixture: dict) -> list[dict]:
    """Team-sheet rows for one fixture, with stat columns attached."""
    source = lineup_source(fixture)
    if source is None:
        return []

    rows = []
    for row in fixture.get("lineups") or []:
        if row.get("type_id") not in (STARTER_TYPE_ID, BENCH_TYPE_ID):
            continue
        # Some historical rows carry player_id: null - a named substitute the
        # provider never resolved to a player record. Dropping them is right:
        # they cannot be linked to any form history, and keeping them would
        # collapse every such player in the cache into one phantom id.
        player_id = row.get("player_id")
        if player_id is None:
            continue

        values = {}
        for detail in row.get("details") or []:
            name = STAT_TYPES.get(detail.get("type_id"))
            if name is None:
                continue
            value = (detail.get("data") or {}).get("value")
            # ints, floats, bools and strings all arrive through this field;
            # keep the numbers and let the rest fall through to the
            # absent-means-zero fill below.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values[name] = value

        rows.append({
            "fixture_id": row.get("fixture_id") or fixture.get("id"),
            "league_id": fixture.get("league_id"),
            "season_id": fixture.get("season_id"),
            "date": fixture.get("starting_at"),
            "team_id": row.get("team_id"),
            "player_id": player_id,
            "is_starting": row.get("type_id") == STARTER_TYPE_ID,
            "position": POSITION_GROUPS.get(row.get("position_id")),
            "lineup_source": source,
            **values,
        })
    return rows


def build_player_table() -> pd.DataFrame:
    rows = []
    paths = sorted(RAW_DIR.glob("*.json.gz"))
    for i, path in enumerate(paths):
        if i and i % 5000 == 0:
            print(f"  {i}/{len(paths)} fixtures, {len(rows)} rows")
        with gzip.open(path, "rt") as fh:
            rows.extend(flatten_fixture(json.load(fh)))

    df = pd.DataFrame(rows)
    for name in STAT_TYPES.values():
        if name not in df.columns:
            df[name] = pd.NA
        if name not in NOT_ZERO_FILLED:
            df[name] = df[name].fillna(0)
        df[name] = pd.to_numeric(df[name], errors="coerce")

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "fixture_id", "team_id"]).reset_index(drop=True)


def build_fixture_sides() -> pd.DataFrame:
    """fixture_id -> (home_id, away_id), from the incumbent's fixture cache."""
    rows = []
    for path in sorted(ALL_FIXTURES_DIR.glob("*.json")):
        fixture = json.loads(path.read_text())
        sides = {
            p["meta"]["location"]: p["id"]
            for p in fixture.get("participants") or []
            if p.get("meta", {}).get("location") in ("home", "away")
        }
        if len(sides) == 2:
            rows.append({"fixture_id": fixture["id"], **{f"{k}_id": v for k, v in sides.items()}})
    return pd.DataFrame(rows)


def main() -> None:
    df = build_player_table()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(df):,} player-match rows to {OUT_PATH}")
    print(f"  {df['fixture_id'].nunique():,} fixtures, {df['player_id'].nunique():,} players")
    print(f"  rating present on {df['rating'].notna().mean():.1%} of rows, "
          f"minutes on {df['minutes_played'].notna().mean():.1%}")

    sides = build_fixture_sides()
    sides.to_parquet(SIDES_PATH, index=False)
    print(f"wrote {len(sides):,} home/away pairs to {SIDES_PATH}")


if __name__ == "__main__":
    main()
