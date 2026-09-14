"""Replay the archive under both form-window scoping recipes, one pass.

Emits one table, one row per fixture that *any* arm in scope_lib.RECIPES
would emit, carrying:

  * two complete sets of form columns, suffixed `__xseason` (the training
    recipe: last MAX_WINDOW matches whatever the season) and `__scoped` (the
    serving recipe: last MAX_WINDOW matches inside the fixture's own season).
    Two sets, not four: `scoped_full8` reads the same columns as `scoped` and
    differs only in which rows it is eligible for, and `xseason_age` reads
    `xseason`'s plus the staleness pair.
  * `home/away_matches_into_season` and `home/away_stale_slots`, the
    `xseason_age` arm's two extra columns per side.
  * `<arm>_eligible` flags, so scope_ablate.py can build both the paired
    intersection and each arm's own coverage without re-deriving the gates.
  * `club_goals_relative` and the league reference, held fixed across arms.
    Those are deliberately *not* season-scoped on either side today
    (db.py:110 says so, and GOALS_WINDOW = 38 is near a full season on
    purpose), so they are not part of the skew and must not move between
    arms - exactly as build_window_features.py holds them fixed.

A scoped window is not a truncation of a cross-season one: team_form()
averages what it is given, so the `__scoped` columns for a club three matches
into a season are a three-match mean, which is the input the service actually
computes and the shape the model has seen on 2% of its training rows.

The `__xseason` columns are asserted to reproduce data/features/
2000_window3_8.csv on every fixture the two share - that check is what makes
this a replay of the training recipe rather than a second pipeline that
resembles it.

Run:
    ./inference-py/bin/python ml/build_scope_features.py      # ~4 min
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_features as bf
import scope_lib
from feature_lib import (
    FORM_FIELD_ORDER,
    LeagueRollingMean,
    club_goals_relative,
    team_form,
)

OUT_DIR = Path(__file__).parent / "data" / "features" / "scope_sweep"
OUT_PATH = OUT_DIR / "form_scope.csv"
CANONICAL_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"

FORM_SOURCES = ("xseason", "scoped")


def build_table() -> pd.DataFrame:
    fixtures = bf.load_fixtures()
    print(f"{len(fixtures)} played fixtures in the archive")

    team_history: dict[int, list[dict]] = {}
    league_goal_volume = LeagueRollingMean()
    rows = []

    for fx in fixtures:
        season_id = fx["season_id"]
        home_hist = team_history.get(fx["home_id"], [])
        away_hist = team_history.get(fx["away_id"], [])
        flags = scope_lib.fixture_eligibility(home_hist, away_hist, season_id)

        if any(flags.values()):
            league_ref = league_goal_volume.reference(fx["league_id"])
            row = {
                "fixture_id": fx["fixture_id"],
                "date": fx["date"],
                "league_id": fx["league_id"],
                "season_id": season_id,
                "result": fx["result"],
                "home_club_goals_relative": club_goals_relative(
                    bf._goal_volume(home_hist), league_ref
                ),
                "away_club_goals_relative": club_goals_relative(
                    bf._goal_volume(away_hist), league_ref
                ),
                **{f"{name}_eligible": value for name, value in flags.items()},
            }
            for side, hist in (("home", home_hist), ("away", away_hist)):
                for source in FORM_SOURCES:
                    w = scope_lib.window(hist, season_id, cross_season=(source == "xseason"))
                    # An arm that is ineligible here still gets columns, from
                    # however short a window exists (possibly empty). Those
                    # cells are never read - the arm's `_eligible` flag drops
                    # the row - but emitting them keeps one rectangular table.
                    form = team_form(w) if w else dict.fromkeys(FORM_FIELD_ORDER, float("nan"))
                    for name, value in form.items():
                        row[f"{side}_form_{name}__{source}"] = value
                for name, value in scope_lib.staleness(hist, season_id).items():
                    row[f"{side}_{name}"] = value
            rows.append(row)

        for side, team_key, gf, ga, stats_key in (
            ("home", "home_id", "home_goals", "away_goals", "home_stats"),
            ("away", "away_id", "away_goals", "home_goals", "away_stats"),
        ):
            team_history.setdefault(fx[team_key], []).append({
                "goals_for": fx[gf],
                "goals_against": fx[ga],
                "stats": fx[stats_key],
                # The only addition to build_features.py's history entry, and
                # the whole reason this replay exists: scoping needs to know
                # which season each prior match belongs to.
                "season_id": season_id,
            })
        league_goal_volume.update(fx["league_id"], bf._goal_volume(team_history[fx["home_id"]]))
        league_goal_volume.update(fx["league_id"], bf._goal_volume(team_history[fx["away_id"]]))

    return pd.DataFrame(rows)


def check_against_canonical(table: pd.DataFrame) -> None:
    """The `__xseason` columns must be the training table where they overlap.

    Only where they overlap, and only on rows `xseason` is eligible for: the
    archive has gained fixtures since 2000_window3_8.csv was written, and this
    table additionally carries rows only the scoped arms can serve. A value
    disagreement on a shared fixture would mean this is not the training
    recipe, which is the thing worth failing on.
    """
    canonical = pd.read_csv(CANONICAL_PATH)
    left = canonical.set_index("fixture_id").sort_index()
    right = table[table["xseason_eligible"]].set_index("fixture_id").sort_index()
    right = right.rename(columns=lambda c: c[: -len("__xseason")]
                         if c.endswith("__xseason") else c)
    cols = [c for c in left.columns if c in right.columns and c != "date"]
    shared = left.index.intersection(right.index)
    pd.testing.assert_frame_equal(
        left.loc[shared, cols], right.loc[shared, cols], check_dtype=False, atol=1e-12
    )
    print(f"__xseason reproduces {CANONICAL_PATH.name} on all {len(cols)} shared columns "
          f"over {len(shared)} shared fixtures "
          f"({len(right) - len(shared)} newer, {len(left) - len(shared)} dropped from it)")


def main() -> None:
    table = build_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Written before the check, so a failing check costs a re-check and not
    # another full replay.
    table.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(table)} rows x {len(table.columns)} cols to {OUT_PATH.name}")
    for name in scope_lib.RECIPES:
        n = int(table[f"{name}_eligible"].sum())
        print(f"  {name:13s} {n:6d} rows  ({n / len(table):6.1%} of the table)")
    check_against_canonical(table)


if __name__ == "__main__":
    main()
