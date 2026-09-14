"""Replays the archive in date order, refitting Dixon-Coles as it goes, and
emits one pre-match score-model row per fixture for every (half-life, cadence)
arm at once.

Same shape as build_rating_features.py, and a separate table joined on
fixture_id for the same reason: the served feature table stays byte-identical,
so a goals arm can be compared against v3 without the form columns having
moved underneath it.

The one thing this file exists to get right is that **a fit only ever sees
matches played before the fixture it prices.** Refits happen on a cadence, and
each refit reads the league's matches strictly before its own date; fixtures
between two refits are priced by the earlier one. A batch model over a 26-year
archive has every opportunity to leak the future, and every number this phase
produces is worthless if it does.

## Pre-registration

Written before the replay was run and before any of its output was read;
`FINDINGS_OPPONENT.md` open item 5 is the precedent - three phases have now
spent their compute confirming what a cheap bound said in advance.

What is already known, from the probe in `FINDINGS_GOALS.md`:
a score matrix over `ratings_lib.AttackDefence`'s *online* rates scores
**0.98673** standalone on the rolling validation set against v3's **0.98497**,
with a draw separation of **+0.0102** against v3's **+0.0119** and the
market's **+0.0176**. So the shape is cheap and the motivating hypothesis -
that a scoreline model picks draws better - is already failing at that fit
quality.

What is untested is the *fit*. These arms solve the time-weighted likelihood
Dixon and Coles describe rather than taking one online gradient step per
fixture. The phase escalates to the rolling ablation if **either**:

1. **Better rates exist.** The best arm's standalone log loss beats the
   online-rate DC baseline of 0.98673 by at least **0.0010**. The probe
   showed two rates coming within 0.0018 of the entire 63-column feature
   table, so a better pair of rates is the highest-leverage object this
   project has left; 0.0010 is ~5% of the 0.0211 market gap and about a
   quarter of the smallest effect the project has ever promoted.

2. **The draw mechanism works after all.** The best arm's draw separation
   reaches v3's +0.0119. That is the hypothesis this phase was ranked on, and
   a batch fit is the only version of it left untested.

If neither holds, the phase stops here and the ablation is not run.

Independently of both, `--diagnose` runs the date-correlation check that
caught three bugs in the ratings phase: any emitted column correlating with
the fixture date beyond |r| = 0.10 is a candidate era proxy, and a tree on a
chronological split will use it for entirely the wrong reason.

That threshold is calibrated for raw columns and is too strict for a fitted
rate, which is why `era_drift` below exists. A smoothed conditional mean has a
fraction of the variance of the outcome it estimates, so a trend that explains
0.4% of the raw variance can explain 3% of the estimate's - the same football,
a much larger r. The check that actually separates the two cases is whether
the column's drift across eras *matches the drift in the thing it estimates*.
Phase 9's career-appearance bug would fail that check; a rate tracking a real
rise in away scoring passes it.

Run:
    ./inference-py/bin/python ml/build_goals_features.py
    ./inference-py/bin/python ml/build_goals_features.py --diagnose
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import goals_lib as gl

FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"
OUT_PATH = FEATURES_DIR / "goals.csv"

#: Half-life of the time weighting, in days. Dixon and Coles fit a decay to
#: their own data and landed near half a season; this brackets that with one
#: arm either side rather than adopting their constant, since their archive
#: was four English divisions in the early nineties.
HALF_LIVES = (180.0, 365.0, 730.0)

#: Days between refits. Swept rather than chosen: a 7-day cadence is the
#: literal paper (every matchday), and the question is whether the fidelity is
#: worth ~13x the refits over an archive this long.
CADENCES = (7.0, 30.0, 90.0)

#: Emitted per arm. The two rates and the three probabilities the score matrix
#: implies - the probabilities are the part a tree cannot construct for itself
#: from the rates, which is the entire argument for handing them over.
FIELDS = ("lam_home", "lam_away", "p_home", "p_draw", "p_away")

#: The arm whose fitted league-level parameters are also emitted, as
#: `dc_rho` and `dc_hfa`. Diagnostics rather than features - nothing in the
#: ablation reads them - and they are here because "what did rho settle at,
#: and did the home advantage drift the way FINDINGS_VENUE.md measured" are
#: questions this phase should be able to answer without a second replay.
DIAGNOSTIC_ARM = (365.0, 30.0)


def arm_name(half_life: float, cadence: float) -> str:
    return f"dc_h{int(half_life)}_c{int(cadence)}"


def arm_columns(half_life: float, cadence: float) -> list[str]:
    return [f"{arm_name(half_life, cadence)}_{field}" for field in FIELDS]


def replay_arm(index: pd.DataFrame, half_life: float, cadence: float) -> np.ndarray:
    """Pre-match (lam, mu, p_home, p_draw, p_away) for every fixture, in the
    row order of `index`.

    One pass per league. Within a league the fixtures are already in date
    order, so the refit schedule is a single forward walk and the window for
    each refit is two searchsorted calls.
    """
    out = np.full((len(index), len(FIELDS)), np.nan)
    # The two fitted league-level parameters, carried alongside so the
    # diagnostic arm can emit them without a second pass.
    league_level = np.full((len(index), 2), np.nan)

    max_age = half_life * np.log2(1.0 / gl.WEIGHT_FLOOR)
    day = np.timedelta64(1, "D")

    for league_id, league in index.groupby("league_id", sort=False):
        dates = league["date"].to_numpy()
        home = league["home_id"].to_numpy()
        away = league["away_id"].to_numpy()
        hg = league["home_goals"].to_numpy(dtype=float)
        ag = league["away_goals"].to_numpy(dtype=float)
        positions = league.index.to_numpy()

        params: gl.LeagueParams | None = None
        next_refit = dates[0]

        for i in range(len(league)):
            today = dates[i]
            if today >= next_refit:
                # Strictly before today, and no older than the weight floor
                # allows. `left` is the first match still carrying weight;
                # `right` is the first match this fit is not allowed to see.
                right = int(np.searchsorted(dates, today, side="left"))
                left = int(np.searchsorted(dates, today - max_age * day, side="left"))
                if right - left >= 2:
                    window = {
                        "days_ago": (today - dates[left:right]) / day,
                        "home_id": home[left:right],
                        "away_id": away[left:right],
                        "home_goals": hg[left:right],
                        "away_goals": ag[left:right],
                    }
                    params = gl.fit_league(window, half_life, warm=params)
                # The schedule advances whether or not there was enough
                # history to fit on, so a league's opening weeks do not queue
                # up a refit per fixture.
                next_refit = today + cadence * day

            if params is None:
                # Before any fit exists, both clubs are unknown and the model
                # is the league-average fixture. Emitted rather than left NaN
                # so the join stays inner and every arm covers the same rows.
                lam = float(np.exp(gl.INIT_HFA))
                mu = 1.0
            else:
                lam, mu = params.rates(int(home[i]), int(away[i]))
            rho = params.rho if params is not None else 0.0
            hfa = params.hfa if params is not None else gl.INIT_HFA
            probs = gl.outcome_probs([lam], [mu], rho)[0]
            out[positions[i]] = (lam, mu, *probs)
            league_level[positions[i]] = (rho, hfa)

    return out, league_level


def build(index: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = index[["fixture_id", "date", "league_id"]].copy()
    timings = {}
    for half_life in HALF_LIVES:
        for cadence in CADENCES:
            started = time.time()
            values, league_level = replay_arm(index, half_life, cadence)
            for column, series in zip(arm_columns(half_life, cadence), values.T):
                frame[column] = series
            if (half_life, cadence) == DIAGNOSTIC_ARM:
                frame["dc_rho"], frame["dc_hfa"] = league_level.T
            timings[arm_name(half_life, cadence)] = time.time() - started
            print(f"  {arm_name(half_life, cadence):16s} "
                  f"{timings[arm_name(half_life, cadence)]:6.1f}s")
    return frame, timings


def diagnose(frame: pd.DataFrame, index: pd.DataFrame) -> None:
    """The pre-registered gate, plus the date-correlation hygiene check.

    Everything here is scored *standalone* - these are the score model's own
    probabilities, with no tree and no form column anywhere near them - which
    is what makes the gate cost seconds rather than the ablation's half hour.
    """
    import rating_ablate as ra
    import train_v2 as tv2

    # The same rows, in the same order, as every rolling-origin number in
    # FINDINGS_VENUE.md and FINDINGS_MARKET.md. Imported rather than
    # reconstructed so a divergence is impossible.
    import venue_ablate as va
    df, _ = va.load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    val = np.concatenate([v for _, v in ra.folds(len(df), "rolling")])
    y_val = y[val]
    fixtures = df["fixture_id"].to_numpy()[val]

    v3 = np.load(FEATURES_DIR / "venue_probs_rolling.npz")["v3"]
    joined = frame.set_index("fixture_id").reindex(fixtures)
    missing = int(joined[arm_columns(*DIAGNOSTIC_ARM)[0]].isna().sum())
    print(f"\n{len(fixtures)} rolling-origin validation fixtures, "
          f"{missing} without a goals row")

    def loss(probs: np.ndarray) -> float:
        return float(-np.log(np.clip(probs[np.arange(len(y_val)), y_val],
                                     1e-15, 1.0)).mean())

    def separation(probs: np.ndarray) -> float:
        return float(probs[y_val == 1, 1].mean() - probs[y_val != 1, 1].mean())

    # --- the two reference points the gate is written against ---------------
    online = gl.outcome_probs(df["exp_goals_home"].to_numpy()[val],
                              df["exp_goals_away"].to_numpy()[val], rho=-0.08)

    print(f"\n{'forecaster':22s} {'log loss':>9s} {'draw sep':>9s} "
          f"{'P(draw)':>9s} {'mean lam_h':>11s} {'mean lam_a':>11s}")
    print("-" * 76)
    print(f"{'v3 (served, 63 cols)':22s} {loss(v3):9.5f} {separation(v3):+9.4f} "
          f"{v3[:, 1].mean():9.4f} {'-':>11s} {'-':>11s}")
    print(f"{'online rates + DC':22s} {loss(online):9.5f} {separation(online):+9.4f} "
          f"{online[:, 1].mean():9.4f} "
          f"{df['exp_goals_home'].to_numpy()[val].mean():11.3f} "
          f"{df['exp_goals_away'].to_numpy()[val].mean():11.3f}")

    scores = {}
    for half_life in HALF_LIVES:
        for cadence in CADENCES:
            name = arm_name(half_life, cadence)
            cols = arm_columns(half_life, cadence)
            probs = joined[cols[2:]].to_numpy()
            scores[name] = (loss(probs), separation(probs))
            print(f"{name:22s} {loss(probs):9.5f} {separation(probs):+9.4f} "
                  f"{probs[:, 1].mean():9.4f} "
                  f"{joined[cols[0]].mean():11.3f} {joined[cols[1]].mean():11.3f}")

    home_goals = index.set_index("fixture_id").reindex(fixtures)
    print(f"\nactual: draw rate {(y_val == 1).mean():.4f}, "
          f"mean goals {home_goals['home_goals'].mean():.3f} home / "
          f"{home_goals['away_goals'].mean():.3f} away")

    # --- the gate ----------------------------------------------------------
    best = min(scores, key=lambda n: scores[n][0])
    best_loss, best_sep = scores[best]
    sharpest = max(scores, key=lambda n: scores[n][1])
    online_loss, v3_sep = loss(online), separation(v3)

    fit_gain = online_loss - best_loss
    print(f"\n{'=' * 76}\nGATE (pre-registered in this file's docstring)\n{'=' * 76}")
    print(f"1. better rates: best arm {best} at {best_loss:.5f} vs the online "
          f"rates' {online_loss:.5f}\n"
          f"   gain {fit_gain:+.5f}, bar +0.00100  ->  "
          f"{'PASS' if fit_gain >= 0.0010 else 'fail'}")
    print(f"2. draw mechanism: sharpest arm {sharpest} separates "
          f"{scores[sharpest][1]:+.4f} vs v3's {v3_sep:+.4f}\n"
          f"   bar: reach v3  ->  "
          f"{'PASS' if scores[sharpest][1] >= v3_sep else 'fail'}")
    passed = fit_gain >= 0.0010 or scores[sharpest][1] >= v3_sep
    print(f"\n=> {'run the ablation' if passed else 'STOP - write the findings'}")

    era_drift(frame, index)


def era_drift(frame: pd.DataFrame, index: pd.DataFrame) -> None:
    """Does each emitted column drift the way the thing it estimates drifts?

    The disambiguation the |r| threshold cannot make. Every column here is an
    estimate of a quantity the archive also records directly, so both can be
    averaged over the same eras and the two drifts compared. A column drifting
    with the football is doing its job; one drifting on its own is the bug.
    """
    merged = frame.merge(index[["fixture_id", "home_goals", "away_goals"]],
                         on="fixture_id", how="inner")
    era = merged["date"].dt.year // 5 * 5
    draw = (merged["home_goals"] == merged["away_goals"]).astype(float)
    home_win = (merged["home_goals"] > merged["away_goals"]).astype(float)
    actual = {
        "lam_home": merged["home_goals"].astype(float),
        "lam_away": merged["away_goals"].astype(float),
        "p_home": home_win,
        "p_draw": draw,
        "p_away": (merged["home_goals"] < merged["away_goals"]).astype(float),
    }

    print(f"\n{'column':22s} {'model first':>12s} {'model last':>11s} "
          f"{'model drift':>12s} {'actual drift':>13s} {'ratio':>7s}")
    print("-" * 82)
    for field in FIELDS:
        column = f"{arm_name(*DIAGNOSTIC_ARM)}_{field}"
        model_by_era = merged.groupby(era)[column].mean()
        actual_by_era = actual[field].groupby(era).mean()
        model_drift = model_by_era.iloc[-1] - model_by_era.iloc[0]
        actual_drift = actual_by_era.iloc[-1] - actual_by_era.iloc[0]
        ratio = model_drift / actual_drift if abs(actual_drift) > 1e-6 else float("nan")
        print(f"{field:22s} {model_by_era.iloc[0]:12.3f} {model_by_era.iloc[-1]:11.3f} "
              f"{model_drift:+12.3f} {actual_drift:+13.3f} {ratio:7.2f}")
    print("A ratio near 1 means the column moves with the football rather than "
          "with the\narchive. Near 0, or the wrong sign, is the era-proxy "
          "failure mode.")

    # The fitted league-level parameters, which are diagnostics and never fed
    # to a model - recorded here because FINDINGS_VENUE.md measured the same
    # decline with a different estimator, and two estimators agreeing is worth
    # more than either alone.
    by_era = merged.groupby(era)[["dc_rho", "dc_hfa"]].mean()
    print(f"\nfitted league parameters by era (diagnostic only, never a feature)")
    print(by_era.round(4).to_string())

    # --- date-correlation hygiene ------------------------------------------
    days = (frame["date"] - frame["date"].min()).dt.days.to_numpy(dtype=float)
    print(f"\n{'column':32s} {'r with date':>12s}")
    print("-" * 46)
    worst = 0.0
    for half_life in HALF_LIVES:
        for cadence in CADENCES:
            for column in arm_columns(half_life, cadence):
                values = frame[column].to_numpy(dtype=float)
                ok = ~np.isnan(values)
                r = float(np.corrcoef(days[ok], values[ok])[0, 1])
                worst = max(worst, abs(r))
                if abs(r) >= 0.10:
                    print(f"{column:32s} {r:+12.3f}   <- era proxy")
    print(f"largest |r| over {len(HALF_LIVES) * len(CADENCES) * len(FIELDS)} "
          f"columns: {worst:.3f}"
          f"  ({'clean' if worst < 0.10 else 'read the era_drift table above'})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    index = pd.read_parquet(INDEX_PATH).sort_values("date").reset_index(drop=True)
    print(f"{len(index)} played fixtures, {index['league_id'].nunique()} leagues, "
          f"{index['date'].min().date()} to {index['date'].max().date()}")

    if OUT_PATH.exists() and args.diagnose:
        frame = pd.read_csv(OUT_PATH, parse_dates=["date"])
        print(f"reusing {OUT_PATH.name}")
    else:
        frame, _ = build(index)
        frame.to_csv(OUT_PATH, index=False)
        print(f"wrote {OUT_PATH} ({len(frame)} rows, {len(frame.columns)} columns)")

    if args.diagnose:
        diagnose(frame, index)


if __name__ == "__main__":
    main()
