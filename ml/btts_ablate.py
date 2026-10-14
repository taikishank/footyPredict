"""Is there a BTTS model worth pursuing, and if so, which of the two obvious
shapes is it?

Two arms answer that, both fitted with no new machinery:

    poisson     goals_lib.btts_prob read off the same Dixon-Coles replay
                build_goals_features.py already runs (half-life 365, cadence
                30 - the file's own DIAGNOSTIC_ARM), via goals_lib.btts_prob.
                No fit of its own: these are the two rates and the low-score
                correction that already exist, pointed at a different
                question than outcome_probs asks of them.
    classifier  train_btts.py's own fit - v3's 63-column feature set and
                recipe, retargeted at the BTTS label. The tree has every
                opponent/form/rating column outcome_probs never sees; the
                question is whether any of that is informative about BTTS
                specifically, or whether it is all 1X2 signal that happens
                not to transfer.

Both are read against `base_rate`, each fold's own training-split BTTS
frequency held constant over its validation slice - the honest null model,
since there is no market line to compare against yet (The Odds API has no
BTTS history to backfill; `ingestor-go` only started collecting it from
whenever that shipped).

`placebo` is `classifier`'s own predictions, shuffled within league after
pooling - same construction as goals_ablate.shuffled() and
recency_ablate.shuffled_weight(), adapted to a probability vector rather
than a feature column. It keeps the arm's predicted distribution and severs
the one thing that would make it informative: which fixture a prediction
belongs to. `classifier` clearing zero against `base_rate` and clearing
`placebo`'s interval too is the same two-sided bar recency_ablate.py's
shuffle arm sets - a gain that does not survive this is a shape artifact,
not a BTTS finding.

Two modes, rating_ablate.folds()'s: `split` is the single chronological
80/20 cut; `rolling` is the five-fold expanding-window protocol every
hyperparameter in this project was actually selected under. Read `split`
for speed while iterating, `rolling` for the number that goes in a findings
file.

Run:
    ./inference-py/bin/python ml/btts_ablate.py --mode split
    ./inference-py/bin/python ml/btts_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import rating_ablate as ra
import train_btts as tb
from build_goals_features import INDEX_PATH, replay_arm
from goals_lib import btts_prob

FEATURES_DIR = Path(__file__).parent / "data" / "features"

BASELINE = "base_rate"
ARMS_ORDER = [BASELINE, "poisson", "classifier", "placebo"]

# The Dixon-Coles arm this phase reads the Poisson probability off -
# build_goals_features.py's own DIAGNOSTIC_ARM, so "poisson" here is backed
# by the same fit that file's era_drift check has already looked at.
POISSON_HALF_LIFE, POISSON_CADENCE = 365.0, 30.0

SHUFFLE_SEED = 0


def poisson_probs(half_life: float, cadence: float) -> pd.DataFrame:
    """Pre-match P(BTTS) for every fixture in the archive, from one
    Dixon-Coles replay.

    Cached because the replay is the expensive part of this phase and
    nothing about it depends on the mode or the classifier arm - it is run
    once per (half_life, cadence) regardless of how many times the rest of
    this script is re-run.
    """
    cache = FEATURES_DIR / f"btts_poisson_h{int(half_life)}_c{int(cadence)}.npz"
    index = pd.read_parquet(INDEX_PATH).sort_values("date").reset_index(drop=True)
    if cache.exists():
        stored = np.load(cache)
        return pd.DataFrame({"fixture_id": stored["fixture_id"],
                             "p_btts_poisson": stored["p_btts"]})

    values, league_level = replay_arm(index, half_life, cadence)
    lam, mu, rho = values[:, 0], values[:, 1], league_level[:, 0]

    # goals_lib.score_matrix takes a scalar rho (`if rho:` cannot branch on an
    # array), so rows are grouped on their fitted rho rather than looped one
    # at a time - rho is piecewise constant between refits, so there are only
    # as many distinct values as there were refits, not one per fixture.
    out = np.empty(len(lam))
    for value in np.unique(rho):
        mask = rho == value
        out[mask] = btts_prob(lam[mask], mu[mask], float(value))

    np.savez(cache, fixture_id=index["fixture_id"].to_numpy(), p_btts=out)
    return pd.DataFrame({"fixture_id": index["fixture_id"].to_numpy(),
                         "p_btts_poisson": out})


def load() -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """train_btts's table (v3's columns + the BTTS label), with the Poisson
    arm's probability joined on as one more column rather than a separate
    frame - same row set, one less thing that can drift apart."""
    df, cols, y = tb.load()
    poisson = poisson_probs(POISSON_HALF_LIFE, POISSON_CADENCE)
    before = len(df)
    df = df.merge(poisson, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, (
        f"{before - len(df)} fixtures have no Dixon-Coles replay row"
    )
    y = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).to_numpy(dtype=int)
    return df, cols, y


def loss_of(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-fixture binary log loss. base_rate/poisson/classifier/placebo all
    go through this one function, so their numbers are comparable by
    construction."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1.0 - p))


def fit_classifier(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                   train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          **tb.PARAMS)
    model.fit(df[cols].iloc[train_idx], y[train_idx])
    return model.predict_proba(df[cols].iloc[val_idx])[:, 1]


def shuffled_within_league(p: np.ndarray, league: np.ndarray, seed: int) -> np.ndarray:
    """`p` permuted within each league - same marginal distribution of
    predictions, no link to which fixture any one of them was actually
    made for. goals_ablate.shuffled()'s construction, applied to a
    probability vector instead of a feature column."""
    rng = np.random.default_rng(seed)
    out = p.copy()
    for value in np.unique(league):
        rows = np.flatnonzero(league == value)
        out[rows] = p[rows[rng.permutation(len(rows))]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    cache = FEATURES_DIR / f"btts_probs_{args.mode}.npz"

    df, cols, y = load()
    league = df["league_id"].to_numpy()
    plan = ra.folds(len(df), args.mode)
    val_idx = np.concatenate([va for _, va in plan])
    y_val, league_val = y[val_idx], league[val_idx]

    print(f"mode={args.mode}, {len(plan)} fold(s), {len(val_idx)} validation "
          f"fixtures, BTTS rate {y_val.mean():.4f}\n")

    probs = {}
    if cache.exists():
        stored = np.load(cache)
        probs = {name: stored[name] for name in stored.files}
        print(f"reusing cached probabilities from {cache.name}\n")

    for name in (BASELINE, "poisson", "classifier"):
        if name in probs:
            continue
        parts = []
        for train_idx, va in plan:
            if name == BASELINE:
                parts.append(np.full(len(va), y[train_idx].mean()))
            elif name == "poisson":
                parts.append(df["p_btts_poisson"].to_numpy()[va])
            else:
                parts.append(fit_classifier(df, cols, y, train_idx, va))
        probs[name] = np.concatenate(parts)
        print(f"  fitted {name:10s}  log loss {loss_of(probs[name], y_val).mean():.5f}")

    if "placebo" not in probs:
        probs["placebo"] = shuffled_within_league(probs["classifier"], league_val,
                                                   SHUFFLE_SEED)
        print(f"  fitted {'placebo':10s}  log loss "
              f"{loss_of(probs['placebo'], y_val).mean():.5f}")
    np.savez(cache, **probs)

    base = loss_of(probs[BASELINE], y_val)
    print(f"\n{'arm':12s} {'log loss':>9s} {'vs base':>9s} {'95% CI':>20s} "
          f"{'P(better)':>10s}")
    print("-" * 64)
    print(f"{BASELINE:12s} {base.mean():9.5f} {'-':>9s} {'-':>20s} {'-':>10s}")
    for name in ARMS_ORDER[1:]:
        stats = ra.paired_bootstrap(base, loss_of(probs[name], y_val))
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:12s} {loss_of(probs[name], y_val).mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")

    print("\nPositive 'vs base' favours the arm. An interval spanning zero is a "
          "null result\nregardless of where the point estimate sits.")

    stats = ra.paired_bootstrap(loss_of(probs["placebo"], y_val),
                                loss_of(probs["classifier"], y_val))
    print(f"\nclassifier against its placebo: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  "
          f"P(real better) = {stats['p_better']:.3f}")
    print("A null here means the classifier's gain over base_rate, if any, is "
          "not distinguishable\nfrom the gain of any same-shaped prediction "
          "vector - the same bar recency_ablate.py's\nshuffle arm sets.")

    stats = ra.paired_bootstrap(loss_of(probs["poisson"], y_val),
                                loss_of(probs["classifier"], y_val))
    print(f"\nclassifier against poisson:     {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  "
          f"P(classifier better) = {stats['p_better']:.3f}")

    print(f"\nvs base_rate, per league:\n\n{'arm':12s}" +
          "".join(f"{value:>10d}" for value in np.unique(league_val)))
    print("-" * (12 + 10 * len(np.unique(league_val))))
    for name in ARMS_ORDER[1:]:
        row = "".join(
            f"{(base[league_val == value] - loss_of(probs[name], y_val)[league_val == value]).mean():>+10.5f}"
            for value in np.unique(league_val)
        )
        print(f"{name:12s}{row}")


if __name__ == "__main__":
    main()
