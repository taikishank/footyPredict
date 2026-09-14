"""Replay the archive at several form-window lengths, one pass.

The incumbent averages a club's last MAX_WINDOW = 8 matches. That number was
never measured - it predates every ablation in this project - and
FINDINGS_PHASE9.md §5 flags it as the cheapest untested lever left: the single
most valuable column in the challenger, club_goals_relative, is club scoring
over *38* matches, and raw-versus-normalised being worth the same says what
carries it is the horizon, not the normalisation.

This emits one feature table per window in WINDOWS, all with:

  * the identical row set - the MIN_WINDOW = 3 gate is held fixed, so a
    fixture is in every table or in none, and the arms can be paired
    fixture-for-fixture by window_ablate.py's bootstrap. Changing the gate
    with the window would confound "longer horizon" with "different fixtures".
  * the identical column names, so arms compose by column selection: a table
    mixing w38 goals with w8 stats needs no extra build.

A short window is *not* a truncation of a long one for a club with little
history - team_form() averages what exists, so a club with five matches gets a
five-match mean under w8 and under w38 alike. The tables diverge only where a
club has more history than the shorter window admits.

The w8 table is asserted to reproduce data/features/2000_window3_8.csv
column-for-column on every fixture the two share, which is what makes this a
sweep of one parameter rather than a second, subtly different pipeline. It is
not row-for-row identical: the raw archive has been refreshed since that table
was written, so these tables carry the fixtures played since as well - the same
fixtures ratings.csv already covers.

Run:
    ./inference-py/bin/python ml/build_window_features.py      # ~8 min
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import build_features as bf
from feature_lib import MAX_WINDOW, MIN_WINDOW, LeagueRollingMean, club_goals_relative, team_form

OUT_DIR = Path(__file__).parent / "data" / "features" / "window_sweep"
CANONICAL_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"

# 4 and 6 ask whether 8 is already too long (form is noisy; recent matches may
# be all that carries). 12/16/24 walk toward club_goals_relative's 38, which
# is included as the far end - the horizon that column says is worth something.
WINDOWS = (4, 6, 8, 12, 16, 24, 38)


def build_tables() -> dict[int, pd.DataFrame]:
    fixtures = bf.load_fixtures()
    print(f"{len(fixtures)} played fixtures in the archive")

    team_history: dict[int, list[dict]] = {}
    league_goal_volume = LeagueRollingMean()
    rows: dict[int, list[dict]] = {w: [] for w in WINDOWS}

    for fx in fixtures:
        home_hist = team_history.get(fx["home_id"], [])
        away_hist = team_history.get(fx["away_id"], [])

        if len(home_hist) >= MIN_WINDOW and len(away_hist) >= MIN_WINDOW:
            league_ref = league_goal_volume.reference(fx["league_id"])
            # club_goals_relative is carried on every table unchanged: it is a
            # separate 38-match quantity, and holding it fixed is what lets
            # window_ablate.py ask whether a longer *form* window is anything
            # the model does not already have through this column.
            shared = {
                "fixture_id": fx["fixture_id"],
                "date": fx["date"],
                "result": fx["result"],
                "home_club_goals_relative": club_goals_relative(
                    bf._goal_volume(home_hist), league_ref
                ),
                "away_club_goals_relative": club_goals_relative(
                    bf._goal_volume(away_hist), league_ref
                ),
            }
            for w in WINDOWS:
                row = dict(shared)
                for name, value in team_form(home_hist, w).items():
                    row[f"home_form_{name}"] = value
                for name, value in team_form(away_hist, w).items():
                    row[f"away_form_{name}"] = value
                rows[w].append(row)

        team_history.setdefault(fx["home_id"], []).append(
            {"goals_for": fx["home_goals"], "goals_against": fx["away_goals"],
             "stats": fx["home_stats"]}
        )
        team_history.setdefault(fx["away_id"], []).append(
            {"goals_for": fx["away_goals"], "goals_against": fx["home_goals"],
             "stats": fx["away_stats"]}
        )
        league_goal_volume.update(fx["league_id"], bf._goal_volume(team_history[fx["home_id"]]))
        league_goal_volume.update(fx["league_id"], bf._goal_volume(team_history[fx["away_id"]]))

    return {w: pd.DataFrame(r) for w, r in rows.items()}


def check_against_canonical(table: pd.DataFrame) -> None:
    """The w8 table must be the served table wherever the two overlap.

    Only where they overlap: the archive has gained fixtures since
    2000_window3_8.csv was written, and those rows are new, not wrong. A value
    disagreement on a *shared* fixture would mean this replay is not the
    served recipe, which is the thing worth failing on.
    """
    canonical = pd.read_csv(CANONICAL_PATH)
    cols = [c for c in canonical.columns if c in table.columns]
    left = canonical[cols].set_index("fixture_id").sort_index()
    right = table[cols].set_index("fixture_id").sort_index()
    shared = left.index.intersection(right.index)
    pd.testing.assert_frame_equal(
        left.loc[shared], right.loc[shared], check_dtype=False, atol=1e-12
    )
    print(f"w{MAX_WINDOW} reproduces {CANONICAL_PATH.name} on all {len(cols)} shared "
          f"columns over {len(shared)} shared fixtures "
          f"({len(right) - len(shared)} newer fixtures not in that table, "
          f"{len(left) - len(shared)} dropped from it)")


def main() -> None:
    tables = build_tables()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Written before the check, so a failing check costs a re-check and not
    # another eight-minute replay.
    for w, table in tables.items():
        out = OUT_DIR / f"form_w{w}.csv"
        table.to_csv(out, index=False)
        print(f"wrote {len(table)} rows x {len(table.columns)} cols to {out.name}")
    check_against_canonical(tables[MAX_WINDOW])


if __name__ == "__main__":
    main()
