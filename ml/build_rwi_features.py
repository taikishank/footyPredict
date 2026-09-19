"""Replays the archive for every arm in rwi_lib.RWI_ARMS and emits one
pre-match row per fixture, joined on fixture_id like ratings.csv.

A third rating table rather than a rebuild of either existing one, for the
reason build_venue_features.py gives: the served v3 rating columns stay
byte-identical, so `v3` sits in the ablation as a reference arm rather than as
a re-fit that might have moved.

Columns are prefixed with the arm name - couple_c25_exp_goal_diff, and so on -
because every arm emits the same seven quantities and they share one table.

Two asserts make the phase readable, one per rule swept:

    bounded_lin  is tanh at a scale where it is the identity, so it must
                 reproduce ratings.csv's atkdef columns
    couple_c0    is CoupledVenue with the coupling off, so it must reproduce
                 venue_ratings.csv's `venue` columns

Between them they say the sweeps differ from the published numbers in their
own parameter and in nothing else.

--diagnose also answers FINDINGS_RWI.md's gating question, which is not about
either arm above: does the archive carry shots, and from when? RWI's third
update-rule idea blends observed goals with xG, and the nearest thing this
archive has is a shot-derived rate. The answer decides whether that arm is
built at all, so it runs before any of this is fitted.

Run:
    ./inference-py/bin/python ml/build_rwi_features.py --diagnose
    ./inference-py/bin/python ml/build_rwi_features.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from build_rating_features import order, season_of
from ratings_lib import ARMS
from rwi_lib import RWI_ARMS
from venue_lib import VenueAttackDefence

RAW_DIR = Path(__file__).parent / "data" / "raw" / "all_fixtures"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"
RATINGS_PATH = FEATURES_DIR / "ratings.csv"
VENUE_PATH = FEATURES_DIR / "venue_ratings.csv"
OUT_PATH = FEATURES_DIR / "rwi_ratings.csv"

#: feature_lib.STAT_IDS entries a shot-derived goal rate would need. Read here
#: rather than imported as a dict so the diagnose says exactly which type ids
#: it looked for.
SHOT_TYPE_IDS = {42: "shots_total", 86: "shots_on_target"}

#: Below this share of a season's fixtures carrying shots for *both* sides,
#: the season cannot contribute a shot-derived update target.
COVERAGE_BAR = 0.90

#: The worst-case per-fixture move build_rating_features.order() documents
#: between the two replay orders, on exp_goal_diff against a column SD of
#: 0.66. The yardstick for couple_c0 against venue_ratings.csv - see
#: check_reproduction().
ORDERING_EFFECT = 1.7


def arm_columns(arm: str) -> list[str]:
    return [f"{arm}_{field}" for field in ARMS["atkdef"].FIELD_ORDER]


def build(index: pd.DataFrame) -> pd.DataFrame:
    """One row per fixture, every arm's pre-match view of it.

    The same walk as build_venue_features.build(): recenter at each league's
    season rollover, regress each club once per season elapsed, features()
    strictly before update() so no row sees its own result.

    The one wrinkle is that this table mixes rating shapes. CoupledVenue keys
    its ratings by (team, slot) and takes a fixture_id; BoundedResidual is a
    plain AttackDefence and does not. Branching on the type keeps both honest
    rather than giving AttackDefence a fixture_id it would ignore.
    """
    arms = {name: factory() for name, factory in RWI_ARMS.items()}
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
            extra = (fx.fixture_id,) if isinstance(arm, VenueAttackDefence) else ()
            values = arm.features(fx.home_id, fx.away_id, fx.league_id, *extra)
            row.update({f"{name}_{k}": v for k, v in values.items()})
        rows.append(row)

        for name, arm in arms.items():
            extra = (fx.fixture_id,) if isinstance(arm, VenueAttackDefence) else ()
            arm.update(fx.home_id, fx.away_id, fx.league_id,
                       fx.home_goals, fx.away_goals, *extra)

        last_season[fx.home_id] = season
        last_season[fx.away_id] = season

    return pd.DataFrame(rows), arms


def _compare(rwi: pd.DataFrame, arm: str, path: Path, prefix: str) -> float | None:
    """Max per-fixture |delta| between an arm's columns and a published table."""
    if not path.exists():
        print(f"{path.name} absent - skipping the {arm} reproduction check")
        return None
    other = pd.read_csv(path)
    merged = rwi.merge(other, on="fixture_id", how="inner", suffixes=("", "_ref"))
    assert len(merged) == len(rwi), f"{path.name} covers different fixtures"
    return max(
        float((merged[f"{arm}_{field}"] - merged[f"{prefix}{field}"]).abs().max())
        for field in ARMS["atkdef"].FIELD_ORDER
    )


def check_reproduction(rwi: pd.DataFrame) -> None:
    """The integrity arms, each against the table it should reproduce.

    Only one of the two can be an assert, and the reason is a defect this
    phase found rather than a compromise. ratings.csv is built under
    build_rating_features.order() - (date, fixture_id), mergesort - which that
    function's docstring explains at length: 68% of fixtures share a kickoff
    timestamp, so `date` alone leaves 27k rows in pandas' quicksort order,
    deterministic given one parquet and derivable from nothing the inference
    service can see.

    build_venue_features.py and build_goals_features.py both sort on `date`
    alone. So venue_ratings.csv is on the unreproducible order, and this
    replay - which is on the correct one, because a rating rule that might be
    served has to be - cannot match both tables at once.

    bounded_lin against ratings.csv is therefore the hard assert, and
    couple_c0 against venue_ratings.csv is reported with the ordering effect
    as its yardstick: order() documents a worst-case move of 1.62 on
    exp_goal_diff, so a delta at that scale is the known ordering difference
    and anything beyond it is a bug in CoupledVenue.
    """
    worst = _compare(rwi, "bounded_lin", RATINGS_PATH, "")
    if worst is not None:
        assert worst < 1e-9, (
            f"bounded_lin does not reproduce ratings.csv's atkdef "
            f"(max |delta| = {worst})")
        print(f"{'bounded_lin':12s} reproduces ratings.csv's atkdef "
              f"(max |delta| = {worst:.2e})")

    worst = _compare(rwi, "couple_c0", VENUE_PATH, "venue_")
    if worst is not None:
        assert worst < ORDERING_EFFECT, (
            f"couple_c0 differs from venue_ratings.csv's venue by {worst}, "
            f"beyond the {ORDERING_EFFECT} the replay ordering explains")
        print(f"{'couple_c0':12s} matches venue_ratings.csv's venue to "
              f"{worst:.3f} - the replay-order\n             difference "
              f"(build_rating_features.order()), not a rating one")


def probe_shot_coverage() -> None:
    """Does the archive carry shots, and from when?

    FINDINGS_RWI.md's gate, and the reason it is a gate rather than a caveat:
    a rating is a *stateful replay from 2000*. A feature with a coverage
    cutoff can be measured on the seasons that have it - that is what
    FINDINGS_PHASE9.md did, at the cost of half the training history. An
    update rule that changes definition partway through carries the
    discontinuity forward in the state itself, into every fixture after it,
    and there is no subset on which it was never applied.
    """
    print(f"reading {RAW_DIR.name}/ for type ids {sorted(SHOT_TYPE_IDS)}...")
    per_season: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "both": 0})

    for path in RAW_DIR.glob("*.json"):
        fixture = json.loads(path.read_text())
        if not fixture.get("starting_at"):
            continue
        sides = {p["meta"]["location"]: p["id"] for p in fixture["participants"]}
        if "home" not in sides or "away" not in sides:
            continue
        have: dict[int, set[int]] = defaultdict(set)
        for entry in fixture.get("statistics") or []:
            if entry["data"].get("value") is not None:
                have[entry["type_id"]].add(entry["participant_id"])
        ids = set(sides.values())
        bucket = per_season[fixture["starting_at"][:4]]
        bucket["n"] += 1
        bucket["both"] += all(ids <= have[t] for t in SHOT_TYPE_IDS)

    print(f"\n{'season':8s} {'fixtures':>9s} {'both sides, both stats':>24s}")
    print("-" * 43)
    usable = total = 0
    for season in sorted(per_season):
        bucket = per_season[season]
        share = bucket["both"] / bucket["n"]
        total += bucket["n"]
        usable += bucket["both"]
        flag = "" if share >= COVERAGE_BAR else "  <-- below bar"
        print(f"{season:8s} {bucket['n']:9d} {share:24.1%}{flag}")
    print(f"\n{'TOTAL':8s} {total:9d} {usable / total:24.1%}")
    print(f"\nThe bar is {COVERAGE_BAR:.0%} of a season's fixtures. A shot-derived "
          "update target\nneeds the seasons *before* the ones it is validated on, "
          "because the rating\nreplays from 2000 - so a cutoff here is not a "
          "training-set cost, it is a\nrule that changes definition mid-state.")


def probe_league_level(arms: dict) -> None:
    """Where each arm's fitted `mu` landed, per league.

    rwi_lib.BoundedResidual's docstring names the risk: a Poisson residual is
    bounded below by -lambda and unbounded above, so compressing the tail
    removes more from the positive side, the ratings drift down, and `mu`
    moves to absorb it. That is not fatal - the rates are what the model
    sees, and they are mu + ratings - but a bound that is really a baseline
    shift should be visible rather than read as a rating effect.
    """
    leagues = sorted({lid for arm in arms.values() for lid in arm.mu})
    print(f"\n{'arm':14s} " + " ".join(f"{lid:>9d}" for lid in leagues))
    print("-" * (14 + 10 * len(leagues)))
    for name, arm in arms.items():
        print(f"{name:14s} " + " ".join(f"{arm.mu.get(lid, float('nan')):9.4f}"
                                        for lid in leagues))
    print("\nA bounded arm whose mu has moved and whose ratings have not is a "
          "baseline\nshift wearing a rating's name.")


def diagnose(rwi: pd.DataFrame, arms: dict) -> None:
    print("=== the gate: shot coverage across the archive ===")
    probe_shot_coverage()
    print("\n=== where each arm's league baseline settled ===")
    probe_league_level(arms)
    print("\n=== era check: does any emitted column just encode the date? ===")
    epoch = rwi["date"].astype("int64")
    for arm in RWI_ARMS:
        for col in arm_columns(arm):
            corr = rwi[col].corr(epoch)
            flag = "  <-- CHECK" if abs(corr) > 0.15 else ""
            print(f"  {col:34s} r(date) = {corr:+.3f}{flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true",
                        help="the gate, and the probes behind this arm set")
    args = parser.parse_args()

    # build_rating_features.order(), not a bare sort_values("date"): 68% of
    # fixtures share a kickoff timestamp, so sorting on the date alone leaves
    # 27k rows in pandas' quicksort order, which is not derivable from
    # anything the inference service can see. ratings.csv is on this order and
    # venue_ratings.csv is not, which is what check_reproduction() below
    # reports on.
    index = order(pd.read_parquet(INDEX_PATH))
    print(f"replaying {len(index)} fixtures over {len(RWI_ARMS)} arms...")
    rwi, arms = build(index)

    check_reproduction(rwi)
    rwi.to_csv(OUT_PATH, index=False)
    total = sum(len(arm_columns(arm)) for arm in RWI_ARMS)
    print(f"wrote {len(rwi)} rows x {total} rating columns to {OUT_PATH}")

    if args.diagnose:
        print()
        diagnose(rwi, arms)


if __name__ == "__main__":
    main()
