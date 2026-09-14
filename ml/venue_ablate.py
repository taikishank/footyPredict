"""Is the served rating's home advantage stale, and is club-level venue
strength real?

FINDINGS_MARKET.md: 61.8% of the gap to Pinnacle's closing line is on away
wins, the model is level with the market on home wins, and the compression is
in discrimination rather than calibration. It named club venue strength as the
fix. build_venue_features.py --diagnose says that club trait does not exist
(split-half reliability -0.06) and that the league's home advantage has fallen
0.30 -> 0.20 on the log scale while AttackDefence carried a near-frozen 0.24.

Every arm here is a drop-in replacement for the served `atkdef` rating, so
every arm carries exactly 63 columns and differs from v3 *only* in the rating
rule:

    hfa_served   AdaptiveHFA at the served rate. Must reproduce v3 exactly;
                 an integrity check on the whole table, not a hypothesis.
    hfa_lr10     the home-advantage update rate x10
    hfa_lr100    x100 - tracks the archive's own log HFA to within 0.01
    hfa_roll     home advantage estimated from the league's last 380
                 fixtures rather than learned
    venue        four ratings per club, split by venue
    placebo      the same four ratings, split at random

The verdict is a paired bootstrap on per-fixture losses, like every phase
before it; rating_ablate.py's machinery is imported rather than restated so
the folds, the hyperparameters and the interval are identical.

Unlike earlier phases the cache holds *probabilities* rather than losses
(FINDINGS_MARKET.md open item 5), so the by-outcome split below - the table
that made this phase's hypothesis - is free rather than a 40-minute re-fit.

Run:
    ./inference-py/bin/python ml/venue_ablate.py --mode split
    ./inference-py/bin/python ml/venue_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import rating_ablate as ra
import train_v2 as tv2
from build_venue_features import arm_columns
from ratings_lib import ARMS
from venue_lib import VENUE_ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"
VENUE_PATH = FEATURES_DIR / "venue_ratings.csv"

ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
BASELINE = "v3"

#: Outcome order matching train_v2.LABEL_MAP, for the by-outcome table.
OUTCOMES = ("home_win", "draw", "away_win")


def load() -> tuple[pd.DataFrame, list[str]]:
    """The v3 table with every arm's rating columns joined on.

    Inner joins throughout, so a fixture missing from any input is missing
    from all arms - the fixture-for-fixture pairing the bootstrap needs.
    """
    df, form = ra.load()
    venue = pd.read_csv(VENUE_PATH).drop(columns=["date", "league_id"])
    before = len(df)
    df = df.merge(venue, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, f"{before - len(df)} fixtures lack venue rating columns"
    return df, form


def arm_plan() -> dict[str, list[str]]:
    """Arm name -> the rating columns it swaps in for atkdef's."""
    plan = {BASELINE: ATKDEF}
    plan.update({name: arm_columns(name) for name in VENUE_ARMS})
    return plan


def fit_probs(df: pd.DataFrame, cols: list[str], y: np.ndarray,
              train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    """Predicted probabilities on the validation slice.

    rating_ablate.fit_and_score's twin, returning the probabilities rather
    than collapsing them to a loss - same model, same hyperparameters, and
    the loss is recovered from these by loss_of() below.
    """
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=tv2.N_ESTIMATORS,
        max_depth=tv2.MAX_DEPTH,
        learning_rate=tv2.LEARNING_RATE,
        reg_lambda=tv2.REG_LAMBDA,
        subsample=tv2.SUBSAMPLE,
        random_state=tv2.RANDOM_STATE,
        eval_metric="mlogloss",
    )
    model.fit(df[cols].iloc[train_idx], y[train_idx])
    return model.predict_proba(df[cols].iloc[val_idx])


def loss_of(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def by_outcome(probs: dict[str, np.ndarray], y: np.ndarray,
               arms: list[str]) -> None:
    """Where each arm's loss sits, in FINDINGS_MARKET.md's own terms.

    That document split the model-market gap by actual outcome and found 61.8%
    of it on away wins and 2.7% on home wins. This is the same split, so a
    fix aimed at away wins can be checked against the table that motivated it
    rather than against the pooled number only.
    """
    print(f"\n{'arm':14s} " + " ".join(f"{o:>10s}" for o in OUTCOMES)
          + f" {'pooled':>10s}")
    print("-" * 62)
    for arm in arms:
        losses = loss_of(probs[arm], y)
        cells = [losses[y == i].mean() for i in range(3)]
        print(f"{arm:14s} " + " ".join(f"{c:10.5f}" for c in cells)
              + f" {losses.mean():10.5f}")

    print(f"\n{'arm':14s} " + " ".join(f"{'P(' + o + ')':>10s}" for o in OUTCOMES))
    print("-" * 48)
    actual = [float((y == i).mean()) for i in range(3)]
    print(f"{'actual':14s} " + " ".join(f"{a:10.4f}" for a in actual))
    for arm in arms:
        mean_p = probs[arm].mean(axis=0)
        print(f"{arm:14s} " + " ".join(f"{p:10.4f}" for p in mean_p))
    print("\nThe served rating's home lean is the row to watch: FINDINGS_MARKET.md "
          "measured\npredicted [0.4522, 0.2444, 0.3033] against actual "
          "[0.4412, 0.2511, 0.3078].")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    cache = FEATURES_DIR / f"venue_probs_{args.mode}.npz"

    df, form_cols = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    val_idx = np.concatenate([v for _, v in plan])
    y_val = y[val_idx]
    arms = arm_plan()
    print(f"mode={args.mode}, {len(plan)} fold(s), {len(val_idx)} validation "
          f"fixtures, {len(arms)} arms\n")

    probs = {}
    if cache.exists():
        stored = np.load(cache)
        probs = {name: stored[name] for name in stored.files}
        print(f"reusing cached probabilities from {cache.name}\n")

    for name, rating_cols in arms.items():
        if name in probs:
            continue
        cols = form_cols + rating_cols
        probs[name] = np.concatenate(
            [fit_probs(df, cols, y, tr, va) for tr, va in plan])
        print(f"  fitted {name:12s} ({len(cols)} cols)  "
              f"log loss {loss_of(probs[name], y_val).mean():.5f}")
    np.savez(cache, **probs)

    base = loss_of(probs[BASELINE], y_val)
    served = loss_of(probs["hfa_served"], y_val)
    delta = float(np.abs(base - served).max())
    print(f"\nintegrity: hfa_served vs v3, max per-fixture |delta| = {delta:.2e}"
          f"  ({'identical' if delta < 1e-9 else 'DIFFERENT - investigate'})")

    print(f"\n{'arm':14s} {'log loss':>9s} {'vs v3':>9s} {'95% CI':>20s} "
          f"{'P(better)':>10s}")
    print("-" * 66)
    print(f"{BASELINE:14s} {base.mean():9.5f} {'-':>9s} {'-':>20s} {'-':>10s}")
    for name in VENUE_ARMS:
        stats = ra.paired_bootstrap(base, loss_of(probs[name], y_val))
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:14s} {loss_of(probs[name], y_val).mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")
    print("\nPositive 'vs v3' favours the arm. An interval spanning zero is a null "
          "result\nregardless of where the point estimate sits.")

    # The control that decides what a positive `venue` would mean. Both arms
    # split each club's ratings in two and halve the evidence behind each; only
    # one of them splits on something football knows.
    v_loss = loss_of(probs["venue"], y_val)
    p_loss = loss_of(probs["placebo"], y_val)
    stats = ra.paired_bootstrap(p_loss, v_loss)
    print(f"\nvenue against its placebo: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  "
          f"P(venue better) = {stats['p_better']:.3f}")
    print("Positive means the venue split beat a random split of the same shape. "
          "An\ninterval through zero means the arm measured its own parameter "
          "count.")

    best = min(VENUE_ARMS, key=lambda n: loss_of(probs[n], y_val).mean())
    print(f"\nbest arm by log loss: {best}. Paired directly against each other arm:")
    print(f"\n{'arm':14s} {'vs ' + best:>12s} {'95% CI':>20s} {'P(best wins)':>13s}")
    print("-" * 62)
    for name in list(VENUE_ARMS) + [BASELINE]:
        if name == best:
            continue
        stats = ra.paired_bootstrap(loss_of(probs[name], y_val), loss_of(probs[best], y_val))
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:14s} {stats['improvement']:+12.5f} {ci:>20s} "
              f"{stats['p_better']:13.3f}")

    by_outcome(probs, y_val, [BASELINE] + list(VENUE_ARMS))


if __name__ == "__main__":
    main()
