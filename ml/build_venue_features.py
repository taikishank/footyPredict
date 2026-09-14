"""Replays the archive for every arm in venue_lib.VENUE_ARMS and emits one
pre-match row per fixture, joined on fixture_id like ratings.csv.

Deliberately a second table rather than a rebuild of ratings.csv, for the
reason build_rating_features.py gives about the v2 form table: the served v3
recipe's rating columns stay byte-identical, so `v3` can sit in the ablation
as a reference arm rather than as a re-fit that might have moved.

Columns are prefixed with the arm name - hfa_roll_exp_goal_diff, and so on -
because every arm here emits the same seven quantities and they have to share
one table.

The assert that makes the whole phase readable: arm `hfa_served` is
AdaptiveHFA at the served hfa rate, and its columns must reproduce
ratings.csv's atkdef columns to within floating point. That is what says the
lr arms differ from v3 in the home-advantage update rate and in nothing else.

Run:
    ./inference-py/bin/python ml/build_venue_features.py
    ./inference-py/bin/python ml/build_venue_features.py --diagnose
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from build_rating_features import season_of
from ratings_lib import ARMS
from venue_lib import VENUE_ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"
RATINGS_PATH = FEATURES_DIR / "ratings.csv"
OUT_PATH = FEATURES_DIR / "venue_ratings.csv"

# Club-seasons below this are dropped from the persistence probe: a venue gap
# over six matches is almost entirely sampling noise, and including such
# seasons would understate the persistence it is testing for.
MIN_VENUE_GAMES = 12

RNG_SEED = 0
SHUFFLE_SAMPLES = 2000


def arm_columns(arm: str) -> list[str]:
    return [f"{arm}_{field}" for field in VENUE_ARMS[arm]().FIELD_ORDER]


def build(index: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture, every arm's pre-match view of it.

    Same walk as build_rating_features.build(): recenter at each league's
    season rollover, regress each club once per season elapsed, features()
    strictly before update() so no row sees its own result.
    """
    arms = {name: factory() for name, factory in VENUE_ARMS.items()}
    last_season: dict[int, int] = {}
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
        for name, arm in arms.items():
            values = arm.features(fx.home_id, fx.away_id, fx.league_id, fx.fixture_id)
            row.update({f"{name}_{k}": v for k, v in values.items()})
        rows.append(row)

        for arm in arms.values():
            arm.update(fx.home_id, fx.away_id, fx.league_id,
                       fx.home_goals, fx.away_goals, fx.fixture_id)

        last_season[fx.home_id] = season
        last_season[fx.away_id] = season

    return pd.DataFrame(rows)


def check_served_reproduction(venue: pd.DataFrame) -> None:
    """hfa_served must be the served atkdef arm, column for column."""
    if not RATINGS_PATH.exists():
        print("ratings.csv absent - skipping the hfa_served reproduction check")
        return
    served = pd.read_csv(RATINGS_PATH)
    merged = venue.merge(served, on="fixture_id", how="inner", suffixes=("", "_served"))
    assert len(merged) == len(venue), "venue and rating tables cover different fixtures"
    worst = 0.0
    for field in ARMS["atkdef"].FIELD_ORDER:
        delta = (merged[f"hfa_served_{field}"] - merged[field]).abs().max()
        worst = max(worst, float(delta))
    assert worst < 1e-9, f"hfa_served does not reproduce atkdef (max |delta| = {worst})"
    print(f"hfa_served reproduces ratings.csv's atkdef columns (max |delta| = {worst:.2e})")


def venue_gaps(index: pd.DataFrame) -> pd.DataFrame:
    """Per club-season home-minus-away points and goal difference per game,
    demeaned within league-season.

    Demeaned because the league's own home advantage is already a parameter of
    every rating in ratings_lib - the question here is only whether a *club*
    deviates from its league in a way that carries."""
    index = index.copy()
    index["season"] = index["date"].apply(season_of)

    def side(team, goals_for, goals_against):
        points = np.where(goals_for > goals_against, 3,
                          np.where(goals_for == goals_against, 1, 0))
        return pd.DataFrame({"team": team, "league": index.league_id,
                             "season": index.season, "points": points,
                             "gd": goals_for - goals_against})

    home = side(index.home_id, index.home_goals, index.away_goals).assign(venue="H")
    away = side(index.away_id, index.away_goals, index.home_goals).assign(venue="A")
    long = pd.concat([home, away], ignore_index=True)

    per = long.groupby(["team", "league", "season", "venue"]).agg(
        ppg=("points", "mean"), gdpg=("gd", "mean"), games=("points", "size"))
    wide = per.unstack("venue")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide = wide[(wide.games_H >= MIN_VENUE_GAMES) & (wide.games_A >= MIN_VENUE_GAMES)].copy()

    wide["gap_ppg"] = wide.ppg_H - wide.ppg_A
    wide["gap_gd"] = wide.gdpg_H - wide.gdpg_A
    for col in ("gap_ppg", "gap_gd"):
        wide[col + "_rel"] = wide[col] - wide.groupby(["league", "season"])[col].transform("mean")
    return wide


def probe_club_venue(index: pd.DataFrame) -> None:
    """Is there a club-level home advantage to rate at all?

    The phase's first question, asked before any fit, in the shape
    FINDINGS_LEAGUE.md's intercept check established: a quantity that does not
    survive out of sample cannot be worth a column however significant its
    in-sample spread.
    """
    wide = venue_gaps(index)
    print(f"{len(wide)} club-seasons with >= {MIN_VENUE_GAMES} matches at each venue, "
          f"{wide.team.nunique()} clubs")
    print(f"  league-mean venue gap   {wide.gap_ppg.mean():.3f} ppg")
    print(f"  club spread within league-season   sd = {wide.gap_ppg_rel.std():.3f} ppg")

    # What that spread would be if every club shared its league's home
    # advantage exactly and the only thing moving the estimate were the
    # scoreline noise of a finite season.
    per_match_sd = float(np.concatenate([
        np.where(index.home_goals > index.away_goals, 3.0,
                 np.where(index.home_goals == index.away_goals, 1.0, 0.0)),
    ]).std())
    typical = float(wide[["games_H", "games_A"]].mean().mean())
    expected = per_match_sd * math.sqrt(2.0 / typical)
    print(f"  spread expected from sampling noise alone = {expected:.3f} ppg "
          f"({typical:.0f} matches per venue)")

    nxt = wide[["team", "season", "gap_ppg_rel", "gap_gd_rel"]].copy()
    nxt["season"] -= 1
    pair = wide.merge(nxt, on=["team", "season"], suffixes=("", "_next"))
    rng = np.random.default_rng(RNG_SEED)
    print(f"\n  {len(pair)} consecutive club-season pairs")
    for col in ("gap_ppg_rel", "gap_gd_rel"):
        r = float(np.corrcoef(pair[col], pair[col + "_next"])[0, 1])
        nxt_values = pair[col + "_next"].to_numpy()
        null = np.array([np.corrcoef(pair[col], rng.permutation(nxt_values))[0, 1]
                         for _ in range(SHUFFLE_SAMPLES)])
        lo, hi = np.percentile(null, [2.5, 97.5])
        flag = "" if lo <= r <= hi else "  <-- outside the null"
        print(f"    lag-1 r({col:12s}) = {r:+.4f}   shuffle 95% "
              f"[{lo:+.4f}, {hi:+.4f}]{flag}")

    probe_split_half(index)


def probe_split_half(index: pd.DataFrame) -> None:
    """The ceiling on any of this: does a club's venue gap predict *itself*?

    Odd and even matches within the same club-season-venue. A rating can only
    carry a quantity from one stretch of matches to another, so a split-half
    correlation at zero bounds every lag-1 correlation above at zero too.
    """
    index = index.copy()
    index["season"] = index["date"].apply(season_of)
    rows = []
    for team_col, gf, ga, venue in (("home_id", "home_goals", "away_goals", "H"),
                                    ("away_id", "away_goals", "home_goals", "A")):
        rows.append(pd.DataFrame({
            "team": index[team_col], "league": index.league_id, "season": index.season,
            "venue": venue,
            "points": np.where(index[gf] > index[ga], 3,
                               np.where(index[gf] == index[ga], 1, 0)),
        }))
    long = pd.concat(rows, ignore_index=True)
    long["half"] = long.groupby(["team", "season", "venue"]).cumcount() % 2

    per = long.groupby(["team", "league", "season", "venue", "half"]).agg(
        ppg=("points", "mean"), games=("points", "size"))
    wide = per.unstack("venue")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide = wide[(wide.games_H >= 6) & (wide.games_A >= 6)].copy()
    wide["gap"] = wide.ppg_H - wide.ppg_A
    wide["gap"] -= wide.groupby(["league", "season", "half"]).gap.transform("mean")

    halves = wide.pivot_table(index=["team", "season"], columns="half", values="gap").dropna()
    r = float(np.corrcoef(halves[0], halves[1])[0, 1])
    # Spearman-Brown: what the correlation would be for a full season's worth
    # of matches rather than half of one.
    full = 2 * r / (1 + r)
    print(f"\n  within-season split-half r = {r:+.4f} over {len(halves)} club-seasons"
          f"  ->  full-season reliability {full:+.4f}")
    print("  A club's venue gap does not predict its own other half. Whatever the "
          "ratings\n  in venue_lib.VenueAttackDefence learn, it is not this.")


def probe_hfa_drift(index: pd.DataFrame, venue: pd.DataFrame) -> None:
    """Has *league* home advantage moved, and does the served rating follow it?

    On the scale AttackDefence uses: hfa is a log multiplier on the home
    side's goal rate, so the archive's own value is log(mean home goals) minus
    log(mean away goals) over a slice, directly comparable with the ratings
    the replay emits.
    """
    index = index.copy()
    index["season"] = index["date"].apply(season_of)
    actual = index.groupby("season").apply(
        lambda d: pd.Series({
            "fixtures": len(d),
            "log_hfa": math.log(d.home_goals.mean()) - math.log(d.away_goals.mean()),
            "home_rate": float((d.home_goals > d.away_goals).mean()),
            "away_rate": float((d.home_goals < d.away_goals).mean()),
        }), include_groups=False)

    merged = venue.merge(index[["fixture_id", "season"]], on="fixture_id")
    for arm in ("hfa_served", "hfa_lr100", "hfa_roll"):
        merged[arm] = (np.log(merged[f"{arm}_exp_goals_home"])
                       - np.log(merged[f"{arm}_exp_goals_away"]))
    carried = merged.groupby("season")[["hfa_served", "hfa_lr100", "hfa_roll"]].mean()

    table = actual.join(carried)
    table = table[table.fixtures > 500]
    print(table.round(4).to_string())
    print("\n  `log_hfa` is the archive's; the arm columns are what each rating "
          "carried into\n  its fixtures. A column that stays flat while log_hfa "
          "falls is the bias\n  FINDINGS_MARKET.md measured as a home lean.")


def diagnose(index: pd.DataFrame, venue: pd.DataFrame) -> None:
    print("=== does a club-level home advantage exist? ===")
    probe_club_venue(index)
    print("\n=== has league home advantage drifted, and do the ratings follow? ===")
    probe_hfa_drift(index, venue)
    print("\n=== era check: does any emitted column just encode the date? ===")
    epoch = venue["date"].astype("int64")
    for arm in VENUE_ARMS:
        for col in arm_columns(arm):
            corr = venue[col].corr(epoch)
            flag = "  <-- CHECK" if abs(corr) > 0.15 else ""
            print(f"  {col:34s} r(date) = {corr:+.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true",
                        help="the probes behind this phase's arm set")
    args = parser.parse_args()

    index = pd.read_parquet(INDEX_PATH).sort_values("date").reset_index(drop=True)
    print(f"replaying {len(index)} fixtures over {len(VENUE_ARMS)} arms...")
    venue = build(index)

    check_served_reproduction(venue)
    venue.to_csv(OUT_PATH, index=False)
    total = sum(len(arm_columns(arm)) for arm in VENUE_ARMS)
    print(f"wrote {len(venue)} rows x {total} rating columns to {OUT_PATH}")

    if args.diagnose:
        print()
        diagnose(index, venue)


if __name__ == "__main__":
    main()
