"""Replays the archive in date order and emits one pre-match rating row per
fixture, for every arm in ratings_lib.ARMS at once.

Deliberately a *separate* table joined on fixture_id rather than a rebuild of
data/features/2000_window3_8.csv. The incumbent's training input stays
byte-identical - the same discipline Phase 9 kept - so a v3 arm can be
compared with the served v2 without the form columns having moved underneath
it, and so this replay can be re-run in seconds while iterating on a rating
rule.

Input is data/features/results_index.parquet: (fixture_id, league_id,
season_id, date, home_id, away_id, home_goals, away_goals) flattened from the
raw archive. Rebuild it with --index if the archive has been refreshed.

Two facts about the archive, established by --diagnose, drive ratings_lib's
shared machinery:

  Five leagues (8/82/301/384/564), zero cross-league fixtures, zero teams in
  more than one league. The rating pool is five disconnected components, so
  ratings are comparable only within a league - which is all that is needed,
  since every fixture to predict is intra-league.

  116 of 211 clubs have a gap season. Without ratings_lib's per-season-elapsed
  regression a club promoted after four years away returns carrying a
  four-year-old rating.

Run:
    ./inference-py/bin/python ml/build_rating_features.py
    ./inference-py/bin/python ml/build_rating_features.py --diagnose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ratings_lib import ARMS

RAW_DIR = Path(__file__).parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"
OUT_PATH = FEATURES_DIR / "ratings.csv"

# A league's season rolls over in midsummer, so the season a fixture belongs
# to is its year minus one for anything before July. Derived from the date
# rather than trusted from season_id because season_id is a provider surrogate
# with no ordering guarantee - subtracting two of them would be meaningless,
# and "how many seasons has this club been away" is exactly a subtraction.
SEASON_CUTOFF_MONTH = 7

# Mirrors build_features.py: the provider carries a final score under this
# type, and a per-participant goals statistic as the fallback for fixtures
# that lack one.
FINAL_SCORE_TYPE_ID = 1525
GOALS_STAT_TYPE_ID = 52


def build_index() -> pd.DataFrame:
    """Flattens the raw archive to the minimal result index the replay needs.

    Deliberately much narrower than build_features.py's parse_fixture(): the
    ratings only ever see who played whom and what the score was, so none of
    the per-team statistics are decompressed. That is what makes the whole
    replay a seconds-long operation once this parquet exists.
    """
    rows = []
    for path in (RAW_DIR / "all_fixtures").glob("*.json"):
        fixture = json.loads(path.read_text())
        try:
            home = next(p for p in fixture["participants"] if p["meta"]["location"] == "home")
            away = next(p for p in fixture["participants"] if p["meta"]["location"] == "away")
        except StopIteration:
            continue

        final = {
            s["score"]["participant"]: s["score"]["goals"]
            for s in fixture["scores"]
            if s["type_id"] == FINAL_SCORE_TYPE_ID
        }
        if "home" in final and "away" in final:
            home_goals, away_goals = final["home"], final["away"]
        else:
            goals = {
                s["participant_id"]: s["data"]["value"]
                for s in fixture["statistics"]
                if s["type_id"] == GOALS_STAT_TYPE_ID
            }
            if home["id"] not in goals or away["id"] not in goals:
                # Unplayed, postponed or cancelled - no result to rate.
                continue
            home_goals, away_goals = goals[home["id"]], goals[away["id"]]

        rows.append((fixture["id"], fixture["league_id"], fixture["season_id"],
                     fixture["starting_at"], home["id"], away["id"],
                     home["name"], away["name"], home_goals, away_goals))

    index = pd.DataFrame(rows, columns=[
        "fixture_id", "league_id", "season_id", "date", "home_id", "away_id",
        "home_name", "away_name", "home_goals", "away_goals",
    ])
    index["date"] = pd.to_datetime(index["date"])
    return index.sort_values("date").reset_index(drop=True)


def season_of(date: pd.Timestamp) -> int:
    return date.year - (1 if date.month < SEASON_CUTOFF_MONTH else 0)


def build(index: pd.DataFrame) -> pd.DataFrame:
    arms = {name: factory() for name, factory in ARMS.items()}
    # Season in which each club was last seen, so a gap can be measured in
    # seasons rather than days. A club that never left has an elapsed of 1 at
    # each rollover and is regressed once, as intended.
    last_season: dict[int, int] = {}
    # Season each league was last seen in, so the re-centering that keeps the
    # level columns from becoming an era proxy fires once per league per
    # season rather than once per fixture.
    league_season: dict[int, int] = {}
    rows = []

    for fx in index.itertuples(index=False):
        season = season_of(fx.date)

        if league_season.get(fx.league_id) != season:
            if fx.league_id in league_season:
                for arm in arms.values():
                    arm.recenter(fx.league_id)
            league_season[fx.league_id] = season

        for team_id in (fx.home_id, fx.away_id):
            previous = last_season.get(team_id)
            if previous is not None and season > previous:
                for arm in arms.values():
                    arm.season_boundary(fx.league_id, team_id, season - previous)

        row = {"fixture_id": fx.fixture_id, "date": fx.date, "league_id": fx.league_id}
        # Strictly before update(): a fixture must never see its own result.
        for arm in arms.values():
            row.update(arm.features(fx.home_id, fx.away_id, fx.league_id))
        rows.append(row)

        for arm in arms.values():
            arm.update(fx.home_id, fx.away_id, fx.league_id, fx.home_goals, fx.away_goals)

        last_season[fx.home_id] = season
        last_season[fx.away_id] = season

    return pd.DataFrame(rows)


def rating_columns() -> list[str]:
    return [field for factory in ARMS.values() for field in factory.FIELD_ORDER]


def diagnose(index: pd.DataFrame, ratings: pd.DataFrame) -> None:
    """The checks that justify ratings_lib's seeding and regression constants,
    printed rather than asserted - they are facts about the archive that a
    future refresh could change."""
    print("=== archive shape ===")
    print(index.groupby("league_id").agg(fixtures=("fixture_id", "size"),
                                         first=("date", "min"), last=("date", "max")))

    teams: dict[int, set[int]] = {}
    seasons: dict[int, set[int]] = {}
    for fx in index.itertuples(index=False):
        for team_id in (fx.home_id, fx.away_id):
            teams.setdefault(team_id, set()).add(fx.league_id)
            seasons.setdefault(team_id, set()).add(season_of(fx.date))
    multi = [t for t, leagues in teams.items() if len(leagues) > 1]
    gapped = [t for t, s in seasons.items()
              if len(s) > 1 and (max(s) - min(s) + 1) != len(s)]
    print(f"\nclubs: {len(teams)}  in >1 league: {len(multi)}  with a gap season: {len(gapped)}")

    print("\n=== is PROMOTED_OFFSET the right sign and size? ===")
    # A club's first season in the archive against the league's incumbents,
    # measured in points per game - the quantity PROMOTED_OFFSET is a prior on.
    first_seen: dict[int, int] = {}
    for fx in index.sort_values("date").itertuples(index=False):
        for team_id in (fx.home_id, fx.away_id):
            first_seen.setdefault(team_id, season_of(fx.date))
    ppg = {"newcomer": [0, 0], "incumbent": [0, 0]}
    for fx in index.itertuples(index=False):
        season = season_of(fx.date)
        for team_id, scored, conceded in (
            (fx.home_id, fx.home_goals, fx.away_goals),
            (fx.away_id, fx.away_goals, fx.home_goals),
        ):
            # Skip each league's founding cohort: they are "new" only because
            # the archive starts, which is not what promotion means.
            if first_seen[team_id] == season_of(index["date"].min()):
                continue
            points = 3 if scored > conceded else (1 if scored == conceded else 0)
            key = "newcomer" if first_seen[team_id] == season else "incumbent"
            ppg[key][0] += points
            ppg[key][1] += 1
    for key, (points, games) in ppg.items():
        print(f"  {key:10s} {points / games:.3f} points/game over {games} games")

    print("\n=== do the ratings separate anything? ===")
    merged = ratings.merge(
        index[["fixture_id", "home_goals", "away_goals"]], on="fixture_id"
    )
    merged["outcome"] = (merged["home_goals"] > merged["away_goals"]).map(
        {True: "home_win", False: "not_home_win"}
    )
    for col in ("elo_diff", "gdelo_diff", "exp_goal_diff"):
        by = merged.groupby("outcome")[col].mean()
        print(f"  {col:16s} home_win {by['home_win']:+8.3f}   "
              f"not {by['not_home_win']:+8.3f}   gap {by['home_win'] - by['not_home_win']:+8.3f}")

    print("\n=== era check: does any rating column just encode the date? ===")
    # The trap Phase 9 hit with career appearance counts - a column that rises
    # with the archive rather than with the football scores well on a
    # chronological split for entirely the wrong reason.
    epoch = ratings["date"].astype("int64")
    for col in rating_columns():
        corr = ratings[col].corr(epoch)
        flag = "  <-- CHECK" if abs(corr) > 0.15 else ""
        print(f"  {col:22s} r(date) = {corr:+.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true",
                        help="print the archive facts behind ratings_lib's constants")
    parser.add_argument("--index", action="store_true",
                        help="rebuild results_index.parquet from the raw archive first "
                             "(~2 min; only needed after the archive is refreshed)")
    args = parser.parse_args()

    if args.index or not INDEX_PATH.exists():
        print(f"building {INDEX_PATH.name} from the raw archive...")
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        build_index().to_parquet(INDEX_PATH, index=False)

    index = pd.read_parquet(INDEX_PATH).sort_values("date").reset_index(drop=True)
    print(f"replaying {len(index)} fixtures over {len(ARMS)} arms...")
    ratings = build(index)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    ratings.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(ratings)} rows x {len(rating_columns())} rating columns to {OUT_PATH}")

    if args.diagnose:
        print()
        diagnose(index, ratings)


if __name__ == "__main__":
    main()
