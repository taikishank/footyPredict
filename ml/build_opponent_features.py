"""Replays the archive and emits, per fixture, the mean opponent ratings each
club's form window saw - plus the two probes ml/FINDINGS_OPPONENT.md
pre-registered as the gate on the ablation.

A second table joined on fixture_id, like ratings.csv and venue_ratings.csv,
for the reason build_venue_features.py gives: the served v3 columns stay
byte-identical, so `v3` sits in the ablation as a reference arm rather than as
a re-fit that might have moved.

Opponent ratings are read from ratings.csv rather than re-derived, so the
strength attributed to a past opponent is exactly the pre-match `atkdef`
value the served model already carries for that fixture. Nothing here can
disagree with the ratings the model is trained on, because it is the same
numbers.

Run:
    ./inference-py/bin/python ml/build_opponent_features.py            # ~6 min
    ./inference-py/bin/python ml/build_opponent_features.py --diagnose
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import build_features as bf
from build_rating_features import season_of
from feature_lib import FORM_FIELD_ORDER, MAX_WINDOW, MIN_WINDOW, team_form
from form_lib import FACED_FIELD_ORDER, OpponentsFaced

FEATURES_DIR = Path(__file__).parent / "data" / "features"
RATINGS_PATH = FEATURES_DIR / "ratings.csv"
OUT_PATH = FEATURES_DIR / "opponent_faced.csv"
PROBE_PATH = FEATURES_DIR / "opponent_probe.parquet"

#: The pre-registered bar, restated here so the code that prints the verdict
#: and the document that fixed it in advance cannot drift apart.
MIN_SCHEDULE_SHARE = 0.05
MIN_QUALIFYING_FIELDS = 5
#: atkdef already carries an opponent-adjusted view of these two, so headroom
#: confined to them restates a column v3 has.
REDUNDANT_FIELDS = ("goals_for", "goals_against")

HOLDOUT_FRACTION = 0.2

RNG_SEED = 0
SHUFFLE_SAMPLES = 200


def load_ratings() -> dict[int, tuple[float, float, float, float]]:
    """fixture_id -> (home_atk, home_def, away_atk, away_def), pre-match."""
    ratings = pd.read_csv(RATINGS_PATH,
                          usecols=["fixture_id", "home_atk", "home_def",
                                   "away_atk", "away_def"])
    return {row.fixture_id: (row.home_atk, row.home_def, row.away_atk, row.away_def)
            for row in ratings.itertuples(index=False)}


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pass producing both outputs.

    The wide table is the feature table the ablation joins on: one row per
    fixture, gated on MIN_WINDOW exactly as build_features.py gates it, so the
    rows line up with 2000_window3_8.csv fixture for fixture.

    The long table is one row per club per match - what that club actually did
    in the match, what its form said beforehand, and who it had been playing -
    which is what the two probes need and which the wide table cannot express.
    """
    fixtures = bf.load_fixtures()
    ratings = load_ratings()
    print(f"{len(fixtures)} played fixtures, {len(ratings)} with rating rows")

    faced = OpponentsFaced()
    team_history: dict[int, list[dict]] = {}
    wide, long = [], []
    skipped = 0

    for fx in fixtures:
        rating = ratings.get(fx["fixture_id"])
        if rating is None:
            # No pre-match rating for this fixture means no strength to
            # attribute to either opponent. Skipping keeps the opponents-faced
            # window honest rather than silently averaging a gap.
            skipped += 1
            continue
        home_atk, home_def, away_atk, away_def = rating

        # build_features.parse_fixture carries starting_at through as the raw
        # string; season_of needs a Timestamp.
        date = pd.Timestamp(fx["date"])
        home_hist = team_history.get(fx["home_id"], [])
        away_hist = team_history.get(fx["away_id"], [])
        home_faced = faced.faced(fx["home_id"])
        away_faced = faced.faced(fx["away_id"])

        if len(home_hist) >= MIN_WINDOW and len(away_hist) >= MIN_WINDOW:
            row = {"fixture_id": fx["fixture_id"], "date": date,
                   "league_id": fx["league_id"]}
            row.update({f"home_{k}": v for k, v in home_faced.items()})
            row.update({f"away_{k}": v for k, v in away_faced.items()})
            wide.append(row)

            # Both sides of the fixture, from that club's own point of view.
            # `opp_atk`/`opp_def` are the opponent it is playing *now* - the
            # thing v3 already knows - and the `_faced` pair is the schedule
            # behind the window, which is the only quantity on trial.
            for club_id, hist, stats, goals_for, goals_against, opp_atk, opp_def, faced_now in (
                (fx["home_id"], home_hist, fx["home_stats"], fx["home_goals"],
                 fx["away_goals"], away_atk, away_def, home_faced),
                (fx["away_id"], away_hist, fx["away_stats"], fx["away_goals"],
                 fx["home_goals"], home_atk, home_def, away_faced),
            ):
                form = team_form(hist)
                actual = dict(stats, goals_for=goals_for, goals_against=goals_against)
                long.append({
                    "fixture_id": fx["fixture_id"],
                    "club_id": club_id,
                    "date": date,
                    "league_season": f'{fx["league_id"]}_{season_of(date)}',
                    "opp_atk": opp_atk,
                    "opp_def": opp_def,
                    **faced_now,
                    **{f"actual_{k}": actual[k] for k in FORM_FIELD_ORDER},
                    **{f"form_{k}": form[k] for k in FORM_FIELD_ORDER},
                })

        team_history.setdefault(fx["home_id"], []).append(
            {"goals_for": fx["home_goals"], "goals_against": fx["away_goals"],
             "stats": fx["home_stats"]})
        team_history.setdefault(fx["away_id"], []).append(
            {"goals_for": fx["away_goals"], "goals_against": fx["home_goals"],
             "stats": fx["away_stats"]})
        faced.record(fx["home_id"], away_atk, away_def)
        faced.record(fx["away_id"], home_atk, home_def)

    if skipped:
        print(f"{skipped} fixtures skipped for want of a rating row")
    return pd.DataFrame(wide), pd.DataFrame(long)


def _demean(frame: pd.DataFrame, cols: list[str], by: pd.Series) -> np.ndarray:
    """Within-league-season demeaning.

    Every quantity here has a level that moved across 26 seasons and differs
    between competitions - possession sums to 100 but shot volume does not,
    and FINDINGS_RATINGS.md's era-proxy bug is the precedent for not letting
    a phase measure the archive's drift by accident. Demeaning inside the
    league-season leaves only within-competition, within-era variation.
    """
    values = frame[cols].to_numpy(dtype=float)
    means = pd.DataFrame(values, index=by.values).groupby(level=0).transform("mean")
    return values - means.to_numpy()


def _ols_r2(x: np.ndarray, y: np.ndarray, train: np.ndarray,
            test: np.ndarray) -> float:
    """Out-of-sample R², intercept included, fitted on `train` only."""
    design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
    resid = y[test] - design[test] @ beta
    total = y[test] - y[train].mean()
    return 1.0 - float(resid @ resid) / float(total @ total)


def probe_schedule_spread(long: pd.DataFrame) -> None:
    """Why the headroom table reads the way it does.

    The mechanism the hypothesis rests on is real and this prints it: a
    per-match R2 of the stat on the opponent actually being played. What the
    form column sees, though, is that mechanism averaged over MAX_WINDOW
    matches, and a league is a round robin - over a full season every club
    faces the same opponents, so schedule strength can only differ in the
    *order* the season deals them out.

    The null is the same move FINDINGS_VENUE.md made on club venue gaps:
    shuffle the opponents within a league-season, keeping each club's match
    count and each league-season's strength distribution, and see how much of
    the observed spread survives as something other than the draw.
    """
    key = long["league_season"]
    rng = np.random.default_rng(RNG_SEED)
    print("Opponent strength, per match and as the form window averages it:\n")
    print(f"{'':14s} {'per match':>10s} {'over ' + str(MAX_WINDOW):>10s} "
          f"{'shuffled':>10s} {'95% of null':>20s}")
    print("-" * 70)

    for field in ("opp_atk", "opp_def"):
        per_match = _demean(long, [field], key)[:, 0]
        observed = _demean(long, [f"{field}_faced"], key)[:, 0].std()

        # Reassign each league-season's opponents at random across its own
        # club-matches, then rebuild each club's rolling window from the
        # shuffled sequence - the same construction as the real column, with
        # the football taken out of who played whom.
        nulls = []
        frame = long[["club_id", "league_season", field]].copy()
        for _ in range(SHUFFLE_SAMPLES):
            frame["shuffled"] = frame.groupby("league_season")[field].transform(
                lambda col: rng.permutation(col.to_numpy()))
            rolled = frame.groupby("club_id")["shuffled"].transform(
                lambda col: col.shift(1).rolling(MAX_WINDOW, min_periods=1).mean())
            nulls.append(_demean(frame.assign(r=rolled).dropna(subset=["r"]),
                                 ["r"], frame.loc[rolled.notna(), "league_season"])[:, 0].std())
        lo, hi = np.percentile(nulls, [2.5, 97.5])
        print(f"{field:14s} {per_match.std():10.4f} {observed:10.4f} "
              f"{np.mean(nulls):10.4f} {f'[{lo:.4f}, {hi:.4f}]':>20s}")

    print(f"\nThe observed spread sits *below* the null, outside its interval. "
          f"A club's last\n{MAX_WINDOW} opponents are more even than {MAX_WINDOW} "
          "drawn at random, because a league is a\nround robin: having played the "
          "strong sides is exactly what leaves the weak\nones to come. So there is "
          "no club-level schedule trait for an adjustment to\nrecover, and less to "
          "recover than chance alone would have left.\n")

    print("The mechanism is real per match, and this is how much of it survives "
          "the\naverage, in each field's own units:\n")
    print(f"{'field':32s} {'per-match R2':>12s} {'correction':>11s} "
          f"{'club spread':>12s} {'ratio':>7s}")
    print("-" * 78)
    now = _demean(long, ["opp_atk", "opp_def"], key)
    faced = _demean(long, list(FACED_FIELD_ORDER), key)
    design = np.column_stack([np.ones(len(now)), now])
    for field in FORM_FIELD_ORDER:
        actual = _demean(long, [f"actual_{field}"], key)[:, 0]
        form = _demean(long, [f"form_{field}"], key)[:, 0]
        if actual.var() == 0 or form.var() == 0:
            continue
        beta, *_ = np.linalg.lstsq(design, actual, rcond=None)
        resid = actual - design @ beta
        correction = (faced @ beta[1:]).std()
        print(f"{field:32s} {1 - resid.var() / actual.var():12.4f} "
              f"{correction:11.3f} {form.std():12.3f} {correction / form.std():7.1%}")
    print("\n'correction' is the sd of the whole opponent adjustment; 'club spread' "
          "the sd\nof the form column it would be applied to. The ratio squared is "
          "the schedule\nshare in the table below.\n")


def probe(long: pd.DataFrame) -> None:
    """The two pre-registered probes, per form field."""
    long = long.sort_values("date").reset_index(drop=True)
    key = long["league_season"]
    faced_cols = list(FACED_FIELD_ORDER)
    now_cols = ["opp_atk", "opp_def"]

    faced = _demean(long, faced_cols, key)
    now = _demean(long, now_cols, key)
    cut = int(len(long) * (1 - HOLDOUT_FRACTION))
    train, test = np.arange(cut), np.arange(cut, len(long))
    print(f"{len(long)} club-matches, {len(test)} in the chronological holdout "
          f"(from {long['date'].iloc[cut].date()})\n")

    print(f"{'field':32s} {'schedule share':>14s} {'dR2 (next match)':>17s}")
    print("-" * 66)
    qualifying = []
    for field in FORM_FIELD_ORDER:
        actual = _demean(long, [f"actual_{field}"], key)[:, 0]
        form = _demean(long, [f"form_{field}"], key)[:, 0]
        if actual.var() == 0 or form.var() == 0:
            continue

        # Probe 1, headroom. Fit what an opponent does to this field on
        # per-match values, then ask how much of the *form column's* variance
        # that mechanism accounts for: the form mean's opponent component is
        # the same coefficients applied to the mean opponent faced.
        design = np.column_stack([np.ones(len(now)), now])
        beta, *_ = np.linalg.lstsq(design, actual, rcond=None)
        schedule = faced @ beta[1:]
        share = float(schedule.var() / form.var())

        # Probe 2, reliability. Both models already know the opponent about to
        # be played, which is what v3 carries; only the schedule behind the
        # window is added.
        base = np.column_stack([form, now])
        r2_base = _ols_r2(base, actual, train, test)
        r2_faced = _ols_r2(np.column_stack([base, faced]), actual, train, test)
        gain = r2_faced - r2_base

        clears = share >= MIN_SCHEDULE_SHARE and gain > 0
        if clears:
            qualifying.append(field)
        print(f"{field:32s} {share:13.1%} {gain:+17.5f}"
              f"{'   <-' if clears else ''}")

    useful = [f for f in qualifying if f not in REDUNDANT_FIELDS]
    print(f"\n{len(qualifying)} field(s) clear both halves of the bar "
          f"({MIN_SCHEDULE_SHARE:.0%} schedule share and a positive dR2); "
          f"{MIN_QUALIFYING_FIELDS} needed.")
    print(f"{len(useful)} of them sit outside {REDUNDANT_FIELDS}, which atkdef "
          f"already carries; 1 needed.")
    passed = len(qualifying) >= MIN_QUALIFYING_FIELDS and len(useful) >= 1
    print(f"\nPRE-REGISTERED VERDICT: {'PROCEED to the ablation' if passed else 'STOP - the ablation is not run'}")
    if qualifying:
        print("qualifying fields: " + ", ".join(qualifying))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    if args.diagnose and PROBE_PATH.exists():
        print(f"reusing {PROBE_PATH.name}\n")
        stored = pd.read_parquet(PROBE_PATH)
        probe_schedule_spread(stored)
        probe(stored)
        return

    wide, long = build()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUT_PATH, index=False)
    long.to_parquet(PROBE_PATH, index=False)
    print(f"wrote {len(wide)} rows to {OUT_PATH}")
    print(f"wrote {len(long)} club-matches to {PROBE_PATH}")
    if args.diagnose:
        print()
        probe_schedule_spread(long)
        probe(long)


if __name__ == "__main__":
    main()
