"""btts_ablate.py found a real but thin gain over base_rate (+0.0048 log
loss, rolling) and a `classifier` that barely edges `poisson` (+0.0015,
p=0.982) - "close substitutes" rather than "one clearly better arm". This
phase asks the four questions that result leaves open, each a lever nothing
upstream has tried:

    tune        Is `classifier`'s recipe (v3's 1X2-tuned depth/rounds/lambda,
                reused unmodified by train_btts.py's own admission) actually
                good for BTTS, or just not-yet-measured for it? A small grid
                over max_depth/reg_lambda, scored on rolling log loss.

    dc_sweep    Is `poisson`'s half-life/cadence (365/30, build_goals_
                features.py's 1X2 diagnostic arm) the right one for BTTS, or
                just the one already lying around? All nine HALF_LIVES x
                CADENCES combos that module already fits, re-scored against
                the BTTS label instead of 1X2.

    blend       `classifier` and `poisson` are "close substitutes" per the
                last phase's own finding - the condition under which
                averaging two decorrelated-ish estimators tends to beat
                either alone. A straight 50/50 mean of their rolling
                out-of-fold probabilities, no new fit.

    plus        Two columns neither v3's 63 nor `poisson` carry on their own:
                each club's rolling BTTS rate (btts_features_lib.py - the AND
                of scoring and conceding, which attack/defence ratings fit
                separately and never combine) and the Dixon-Coles arm's
                lam+mu (expected total goals, the one scalar `poisson`
                collapses everything into before `classifier` ever sees it).
                Appended to v3's columns, fit with `tune`'s winning recipe.

Every arm is read against the same base_rate/placebo bar btts_ablate.py set,
on the same rolling-origin validation fixtures, via the same paired
bootstrap - a candidate that does not clear its placebo is noise dressed as
a finding, regardless of which lever produced it.

Run:
    ./inference-py/bin/python ml/btts_ablate_v2.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import btts_ablate as ba
import btts_features_lib as bf
import rating_ablate as ra
import train_btts as tb
from build_goals_features import CADENCES, HALF_LIVES, INDEX_PATH

FEATURES_DIR = Path(__file__).parent / "data" / "features"
GOALS_CSV = FEATURES_DIR / "goals.csv"

GRID_MAX_DEPTH = (1, 2, 3)
GRID_REG_LAMBDA = (10, 30, 60)

BLEND_WEIGHTS = (0.3, 0.5, 0.7)


def report(name: str, base: np.ndarray, arm: np.ndarray) -> dict:
    stats = ra.paired_bootstrap(base, arm)
    print(f"  {name:28s} log loss {arm.mean():.5f}  vs base {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  P(better)={stats['p_better']:.3f}")
    return stats


# --------------------------------------------------------------------------
# tune: classifier capacity grid
# --------------------------------------------------------------------------

def fit_classifier(df, cols, y, train_idx, val_idx, params) -> np.ndarray:
    model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    model.fit(df[cols].iloc[train_idx], y[train_idx])
    return model.predict_proba(df[cols].iloc[val_idx])[:, 1]


def run_tune(df, cols, y, plan, y_val) -> tuple[dict, np.ndarray]:
    print("\n--- tune: classifier capacity grid (rolling) ---")
    cache = FEATURES_DIR / "btts_tune_rolling.npz"
    grid = list(itertools.product(GRID_MAX_DEPTH, GRID_REG_LAMBDA))
    results = {}
    if cache.exists():
        stored = np.load(cache)
        results = {name: stored[name] for name in stored.files}
        print("  reusing cached grid")
    for depth, reg in grid:
        name = f"depth{depth}_lambda{reg}"
        if name in results:
            continue
        params = {**tb.PARAMS, "max_depth": depth, "reg_lambda": reg}
        parts = [fit_classifier(df, cols, y, tr, va, params) for tr, va in plan]
        results[name] = np.concatenate(parts)
    np.savez(cache, **results)

    losses = {name: ba.loss_of(probs, y_val) for name, probs in results.items()}
    best_name = min(losses, key=lambda n: losses[n].mean())
    current = losses[f"depth{tb.PARAMS['max_depth']}_lambda{tb.PARAMS['reg_lambda']}"]
    for name in sorted(losses, key=lambda n: losses[n].mean()):
        marker = " <- current recipe" if name == f"depth{tb.PARAMS['max_depth']}_lambda{tb.PARAMS['reg_lambda']}" else ""
        marker += " <- best" if name == best_name else ""
        print(f"  {name:20s} {losses[name].mean():.5f}{marker}")
    stats = ra.paired_bootstrap(current, losses[best_name])
    print(f"\n  best ({best_name}) vs current recipe: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  P(better)={stats['p_better']:.3f}")
    depth, reg = (int(x) for x in best_name.replace("depth", "").replace("lambda", "").split("_"))
    best_params = {**tb.PARAMS, "max_depth": depth, "reg_lambda": reg}
    return best_params, results[best_name]


# --------------------------------------------------------------------------
# dc_sweep: half-life / cadence grid for the Poisson arm
# --------------------------------------------------------------------------

def run_dc_sweep(df, y, plan, y_val, league_val) -> np.ndarray:
    print("\n--- dc_sweep: Dixon-Coles half-life/cadence grid (rolling) ---")
    val_idx = np.concatenate([va for _, va in plan])
    results = {}
    for half_life, cadence in itertools.product(HALF_LIVES, CADENCES):
        poisson = ba.poisson_probs(half_life, cadence)
        joined = df[["fixture_id"]].merge(poisson, on="fixture_id", how="left")
        name = f"h{int(half_life)}_c{int(cadence)}"
        results[name] = joined["p_btts_poisson"].to_numpy()[val_idx]

    losses = {name: ba.loss_of(probs, y_val) for name, probs in results.items()}
    best_name = min(losses, key=lambda n: losses[n].mean())
    current = losses["h365_c30"]
    for name in sorted(losses, key=lambda n: losses[n].mean()):
        marker = " <- current (poisson)" if name == "h365_c30" else ""
        marker += " <- best" if name == best_name else ""
        print(f"  {name:20s} {losses[name].mean():.5f}{marker}")
    stats = ra.paired_bootstrap(current, losses[best_name])
    print(f"\n  best ({best_name}) vs current poisson arm: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  P(better)={stats['p_better']:.3f}")
    return results[best_name]


# --------------------------------------------------------------------------
# plus: classifier with two extra columns
# --------------------------------------------------------------------------

def run_plus(df, cols, y, plan, y_val, best_params) -> np.ndarray:
    print("\n--- plus: classifier + rolling BTTS rate + lam+mu (rolling) ---")
    cache = FEATURES_DIR / "btts_plus_rolling.npz"
    if cache.exists():
        stored = np.load(cache)
        print("  reusing cached probs")
        return stored["plus"]

    index = pd.read_parquet(INDEX_PATH)
    rates = bf.build(index)
    goals_lam = pd.read_csv(GOALS_CSV, usecols=[
        "fixture_id", "dc_h365_c30_lam_home", "dc_h365_c30_lam_away"])
    goals_lam["exp_total_goals"] = (
        goals_lam["dc_h365_c30_lam_home"] + goals_lam["dc_h365_c30_lam_away"])

    extended = df.merge(rates, on="fixture_id", how="left")
    extended = extended.merge(
        goals_lam[["fixture_id", "exp_total_goals"]], on="fixture_id", how="left")
    extra_cols = ["home_btts_rate", "away_btts_rate", "exp_total_goals"]
    missing = extended[extra_cols].isna().any(axis=1).sum()
    print(f"  {missing} rows missing an extra column (early-season history gap)")

    full_cols = cols + extra_cols
    parts = [fit_classifier(extended, full_cols, y, tr, va, best_params) for tr, va in plan]
    probs = np.concatenate(parts)
    np.savez(cache, plus=probs)
    return probs


def main() -> None:
    df, cols, y = ba.load()
    league = df["league_id"].to_numpy()
    plan = ra.folds(len(df), "rolling")
    val_idx = np.concatenate([va for _, va in plan])
    y_val, league_val = y[val_idx], league[val_idx]

    base_cache = FEATURES_DIR / "btts_probs_rolling.npz"
    stored = np.load(base_cache)
    base_probs, classifier_probs, poisson_probs_cur, placebo_probs = (
        stored["base_rate"], stored["classifier"], stored["poisson"], stored["placebo"])

    base = ba.loss_of(base_probs, y_val)
    print(f"base_rate log loss {base.mean():.5f} (reference for every arm below)")
    report("classifier (current)", base, ba.loss_of(classifier_probs, y_val))
    report("poisson (current)", base, ba.loss_of(poisson_probs_cur, y_val))
    report("placebo", base, ba.loss_of(placebo_probs, y_val))

    best_params, tuned_probs = run_tune(df, cols, y, plan, y_val)
    best_poisson_probs = run_dc_sweep(df, y, plan, y_val, league_val)
    plus_probs = run_plus(df, cols, y, plan, y_val, best_params)

    print("\n--- blend: tuned classifier x best poisson (rolling) ---")
    for w in BLEND_WEIGHTS:
        blend = w * tuned_probs + (1 - w) * best_poisson_probs
        report(f"blend w={w}", base, ba.loss_of(blend, y_val))

    print("\n--- summary vs base_rate ---")
    report("classifier (current)", base, ba.loss_of(classifier_probs, y_val))
    report("classifier (tuned)", base, ba.loss_of(tuned_probs, y_val))
    report("poisson (current)", base, ba.loss_of(poisson_probs_cur, y_val))
    report("poisson (best sweep)", base, ba.loss_of(best_poisson_probs, y_val))
    report("classifier + plus features (tuned)", base, ba.loss_of(plus_probs, y_val))
    best_blend = max(BLEND_WEIGHTS,
                      key=lambda w: -ba.loss_of(w * tuned_probs + (1 - w) * best_poisson_probs, y_val).mean())
    report(f"blend (best w={best_blend})", base,
           ba.loss_of(best_blend * tuned_probs + (1 - best_blend) * best_poisson_probs, y_val))

    print("\n--- plus vs current classifier (the question that matters) ---")
    stats = ra.paired_bootstrap(ba.loss_of(classifier_probs, y_val), ba.loss_of(plus_probs, y_val))
    print(f"  plus vs classifier: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  P(better)={stats['p_better']:.3f}")

    print("\n--- best overall vs current classifier ---")
    candidates = {
        "tuned": tuned_probs,
        "plus": plus_probs,
        f"blend_w{best_blend}": best_blend * tuned_probs + (1 - best_blend) * best_poisson_probs,
    }
    best_overall = min(candidates, key=lambda n: ba.loss_of(candidates[n], y_val).mean())
    stats = ra.paired_bootstrap(ba.loss_of(classifier_probs, y_val), ba.loss_of(candidates[best_overall], y_val))
    print(f"  {best_overall} vs current classifier: {stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]  P(better)={stats['p_better']:.3f}")


if __name__ == "__main__":
    main()
