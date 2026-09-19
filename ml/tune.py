"""Hyperparameter sweeps for the match-outcome models, logged to MLflow as one
parent run with a nested child run per grid point.

Covers three models (--model v1|v2|v3), differing only in which columns they
read and which recipe they are measured against:

    v1  team form only                      -> experiment liveedge-match-outcome
    v2  team form + club_goals_relative      -> experiment liveedge-match-outcome-v2
    v3  v2 + atkdef ratings (63 columns)     -> experiment liveedge-match-outcome-v3

v1 and v2 share one feature table; v3 joins ml/data/features/ratings.csv onto it
through rating_ablate.load(), so its rows are the same 39,594 every number in
FINDINGS_RATINGS.md onward was measured on.

--mode picks the validation protocol. `split` is the single chronological
holdout this script has always used; `rolling` is rating_ablate.py's five
expanding-window folds, which is what every finding since FINDINGS_RATINGS.md
reports. The one-fold case *is* the split, so the two share a code path and a
`--mode split` run reproduces what this script did before rolling origin
existed.

Running both is deliberate and they answer different questions. `rolling`
decides. `split` is a control: this project has used `--mode split` as a cheap
pre-check in every ablation phase and has never measured whether that habit
survives small differences, so a full grid under both protocols reports the
rank agreement between them as a result in its own right.

Sweeping v1 on the same grid as v2 is what makes a v1-vs-v2 comparison
meaningful after v2 has been tuned. ml/train_v2.py deliberately mirrored
ml/train.py's hyperparameters so that any validation difference was
attributable to v2's two extra columns alone; the moment v2 is tuned and v1 is
not, that attribution breaks and "v2 is better" stops distinguishing a better
feature from a better fit. Running the identical grid against both restores it.

Two things this does that eyeballing `log_loss` across runs does not:

1. Paired significance. Two configs' mean log losses can differ in the 4th
   decimal and mean nothing - the val set's per-match log loss has a large
   spread, so the mean carries real sampling error. What matters is the
   *paired* per-match difference against a reference config, whose standard
   error is much smaller because the same matches are being compared. Each
   child run logs that difference, its SE, and a t-statistic.

2. No artifact clobbering. Nothing is written to either model's local path or
   to S3 - the served models are left alone. The single best config's model is
   logged to MLflow at the end, so promoting it later means loading that exact
   artifact rather than retraining and trusting the fit to reproduce.

Run:
    ./inference-py/bin/python ml/tune.py --model v2 --grid reg
    ./inference-py/bin/python ml/tune.py --model v1 --grid reg
    ./inference-py/bin/python ml/tune.py --model v1 --grid reg --dry-run
    ./inference-py/bin/python ml/tune.py --model v3 --grid cap --mode split
    ./inference-py/bin/python ml/tune.py --model v3 --grid cap --mode rolling
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import rating_ablate as ra  # noqa: E402
import train_v2 as tv2  # noqa: E402

FEATURES_PATH = Path(__file__).parent / "data" / "features" / "2000_window3_8.csv"
MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"

#: Per-point results, one CSV per (model, grid, mode). Written so a later run
#: under the *other* protocol can report rank agreement against it without
#: re-fitting anything.
RESULTS_DIR = Path(__file__).parent / "data" / "tune"

# The split fraction now comes from rating_ablate.folds(), which reads
# train_v2.VAL_FRACTION - the same 0.2 this script used to hold itself. Stating
# it once means a --mode split run here and a --mode split ablation cut the
# archive in the same place.
LABEL_MAP = {"home_win": 0, "draw": 1, "away_win": 2}

# v1 and v2 start from this recipe (ml/train.py and ml/train_v2.py carry the
# same three constants), so it is the paired reference for either sweep. Note
# what it leaves out: the regularization constants train_v2.py later gained.
# That was the point for those sweeps - they were measuring what regularization
# was worth against an unregularized reference.
BASELINE = {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.0072}

# v3's reference is the *served* recipe instead, regularization included, so a
# v3 sweep is measured against the 0.98497 that FINDINGS_RATINGS.md onward
# publishes rather than against a config nothing has ever run. The constants are
# imported from train_v2 rather than restated, for the reason rating_ablate.py's
# fit_and_score gives: a restated constant silently drifting is a confound that
# reads as a finding.
SERVED_V3 = {**BASELINE, "reg_lambda": tv2.REG_LAMBDA, "subsample": tv2.SUBSAMPLE}

# Which experiment each sweep lands in, matching the training script that owns
# the model, so a sweep sits alongside that model's own runs rather than in a
# third place. The extra v2 columns are named here rather than in the grids
# because they are a property of the model, not of the sweep; v3's rating
# columns come from rating_ablate's join instead, which is why its list is
# empty here.
MODELS = {
    "v1": {"experiment": "liveedge-match-outcome", "extra_cols": [],
           "baseline": BASELINE},
    "v2": {
        "experiment": "liveedge-match-outcome-v2",
        "extra_cols": ["home_club_goals_relative", "away_club_goals_relative"],
        "baseline": BASELINE,
    },
    "v3": {
        "experiment": "liveedge-match-outcome-v3",
        "extra_cols": [],
        "baseline": SERVED_V3,
    },
}

GRIDS = {
    "core": {
        "n_estimators": [400, 800, 1600],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.003, 0.0072, 0.02, 0.05],
    },
    "reg": {
        "min_child_weight": [1, 5, 20],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [1, 10, 100],
    },
    # The "reg" sweep pinned its optimum to the grid's edge on reg_lambda
    # (monotone 1 -> 10 -> 100) and subsample (monotone 1.0 -> 0.7), so the
    # real optimum was outside it. This pushes both past where "reg" stopped,
    # and found reg_lambda=100 to be a true interior optimum after all.
    "reg2": {
        "subsample": [0.5, 0.6, 0.7, 0.85],
        "colsample_bytree": [0.6, 1.0],
        "reg_lambda": [100, 300, 1000, 3000],
    },
    # The capacity sweep, and the reason this script gained a v3 model at all.
    #
    # Two placebos have now cleared zero against the served v3 config:
    # FINDINGS_LEAGUE.md's twenty random-partition interactions (+0.00074
    # [+0.00034, +0.00113]) and FINDINGS_GOALS.md's three shuffled Poisson
    # rates (+0.00032 [+0.00003, +0.00062]). Two unrelated noise injections of
    # different shapes both improving a model significantly is the signature of
    # one that cannot reach the interactions its feature set already contains.
    #
    # Both grids that produced the served config were selected on v2's 56
    # columns before atkdef existed, and neither goes where that hypothesis
    # points: `core` tops out at max_depth 5, and `reg2` pushed reg_lambda
    # *upward* from 100 to 3000. This one extends toward MORE capacity and
    # brackets the served config on both sides of both axes, so its optimum can
    # be read as interior or as pinned to an edge:
    #
    #     max_depth   3 <- [4 served] -> 5, 6, 8
    #     reg_lambda  10, 30 <- [100 served] -> 300
    #
    # subsample stays at the served 0.6 and is not swept: reg2 found its curve
    # flat between 0.5 and 0.7, and a third axis would triple a grid whose cost
    # is already dominated by max_depth 8 (4x a depth-4 fit).
    "cap": {
        "max_depth": [3, 4, 5, 6, 8],
        "reg_lambda": [10, 30, 100, 300],
    },
    # `cap` answered its question and inverted it. Depth is monotone on both
    # protocols - 3:0.98390  4:0.98535  5:0.98706  6:0.98877  8:0.99217 under
    # rolling origin - so v3 wants LESS tree than it is served with, not more,
    # and the under-capacity reading of the two placebos is dead. But the
    # winner sat on the grid's edge at max_depth 3, which is the same defect
    # `reg` had when its optimum pinned at reg_lambda 100 and needed `reg2`.
    #
    # This extends downward until the optimum is interior. It also picks up
    # n_estimators, because the two are not separable: dropping a level of
    # depth removes capacity that a longer boost at the same learning rate can
    # put back, and a shallower tree that wants 2400 rounds is a different
    # finding from one that wants 800. reg_lambda drops 300 (worst cell on both
    # protocols) and keeps the three that bracket rolling's optimum at 30.
    "cap2": {
        "max_depth": [1, 2, 3],
        "reg_lambda": [10, 30, 100],
        "n_estimators": [800, 2400],
    },
}

# Fixed so a rerun of the same config reproduces. The training scripts leave
# this at XGBoost's default, which is also 0, so the baseline row here matches
# what they produce.
RANDOM_STATE = 0

SIGNIFICANCE_T = 1.96  # two-sided 95%


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def per_match_nll(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Log loss per row, so two configs can be compared match-by-match."""
    return -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-15, None))


def load_table(model: str) -> tuple[pd.DataFrame, list[str]]:
    """The feature table and column list for one model.

    v3 goes through rating_ablate.load() rather than reading the CSV directly,
    so its rows are the inner join of the form table with ratings.csv - exactly
    the 39,594 fixtures FINDINGS_RATINGS.md onward scores. Reproducing that row
    set is what lets a v3 sweep's baseline point be checked against the
    published 0.98497 instead of taken on trust.
    """
    if model == "v3":
        df, base_cols = ra.load()
        return df, base_cols + ra.ARM_COLUMNS["atkdef"]

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    form_cols = [c for c in df.columns if c.startswith(("home_form_", "away_form_"))]
    return df, form_cols + MODELS[model]["extra_cols"]


def evaluate(params: dict, df: pd.DataFrame, cols: list[str], y: np.ndarray,
             fold_list: list) -> tuple[np.ndarray, XGBClassifier]:
    """Validation probabilities for one config, concatenated across folds.

    Returns the last fold's fitted model alongside them, which is the one worth
    logging: under `rolling` it is the fold trained on the most history, and
    under `split` it is the only fit there was.
    """
    parts, model = [], None
    for train_idx, val_idx in fold_list:
        # random_state goes in through the same mapping rather than as its own
        # keyword, so a caller sweeping seeds (ml/seed_probe.py) can override it
        # without this function growing an argument for it.
        model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            **{"random_state": RANDOM_STATE, **params},
        )
        model.fit(df[cols].iloc[train_idx], y[train_idx])
        parts.append(model.predict_proba(df[cols].iloc[val_idx]))
    return np.vstack(parts), model


def rank_agreement(res: pd.DataFrame, varied: list[str], model: str, grid: str,
                   mode: str) -> None:
    """Compare this run's ranking against the same grid under the other protocol.

    The control described in the module docstring. Prints nothing useful if the
    other run has not happened yet, so the two can be run in either order.
    """
    other_mode = "split" if mode == "rolling" else "rolling"
    path = RESULTS_DIR / f"{model}_{grid}_{other_mode}.csv"
    if not path.exists():
        print(f"\n(no {other_mode} run to compare against yet - "
              f"run --mode {other_mode} for the protocol control)")
        return

    other = pd.read_csv(path)
    merged = res.merge(other, on=varied, suffixes=("", "_other"))
    if len(merged) != len(res):
        print(f"\nWARNING: only {len(merged)}/{len(res)} grid points matched "
              f"{path.name}; the two runs swept different grids")
    if merged.empty:
        return

    rho = merged["log_loss"].rank().corr(merged["log_loss_other"].rank())
    this_best = merged.sort_values("log_loss").iloc[0]
    other_order = merged.sort_values("log_loss_other").reset_index(drop=True)
    where = int(other_order[
        (other_order[varied] == this_best[varied]).all(axis=1)
    ].index[0]) + 1

    print(f"\nPROTOCOL CONTROL - this ({mode}) against {path.name}:")
    print(f"  Spearman rank correlation over {len(merged)} points: {rho:+.3f}")
    print(f"  this run's winner ranks {where}/{len(merged)} under {other_mode}")
    if mode == "rolling":
        verdict = ("a top-3 shortlist off the split would have kept it"
                   if where <= 3 else
                   "a top-3 shortlist off the split would have DISCARDED it")
        print(f"  -> {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), default="v2",
                    help="which feature set to sweep (default: v2)")
    ap.add_argument("--grid", choices=sorted(GRIDS), default="core",
                    help="which named sweep to run (default: core)")
    ap.add_argument("--mode", choices=("split", "rolling"), default="split",
                    help="validation protocol (default: split, this script's original)")
    ap.add_argument("--dry-run", action="store_true", help="list the grid, fit nothing")
    args = ap.parse_args()

    spec = MODELS[args.model]
    baseline = spec["baseline"]
    grid = GRIDS[args.grid]
    combos = [
        {**baseline, **dict(zip(grid.keys(), values))}
        for values in itertools.product(*grid.values())
    ]
    varied = list(grid.keys())

    if args.dry_run:
        for i, c in enumerate(combos, 1):
            print(f"{i:3d}. " + "  ".join(f"{k}={c[k]}" for k in varied))
        print(f"\n{len(combos)} grid point(s) in '{args.grid}' for model {args.model}; "
              f"held fixed at {baseline}")
        return

    df, feature_cols = load_table(args.model)
    y = df["result"].map(LABEL_MAP).to_numpy()
    fold_list = ra.folds(len(df), args.mode)
    val_all = np.concatenate([val for _, val in fold_list])
    y_val = y[val_all]

    print(f"model {args.model} -> experiment '{spec['experiment']}'")
    print(f"protocol {args.mode}: {len(fold_list)} fold(s), "
          f"{len(val_all)} validation rows of {len(df)}")
    for i, (train_idx, val_idx) in enumerate(fold_list, 1):
        print(f"  fold {i}: train {len(train_idx):6d} rows, val {len(val_idx):5d} "
              f"({df['date'].iloc[val_idx[0]].date()} .. "
              f"{df['date'].iloc[val_idx[-1]].date()})")
    print(f"{len(feature_cols)} features, {len(combos)} point(s) in grid '{args.grid}'")
    print(f"varying {varied}, holding {baseline}\n")

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(spec["experiment"])

    run_name = f"sweep-{args.model}-{args.grid}-{args.mode}-{len(combos)}pt"
    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params({
            "sweep": args.grid,
            "model": args.model,
            "mode": args.mode,
            "n_folds": len(fold_list),
            "n_grid_points": len(combos),
            "n_features": len(feature_cols),
            "n_rows": len(df),
            "n_val_rows": len(val_all),
            "random_state": RANDOM_STATE,
            **{f"baseline_{k}": v for k, v in baseline.items()},
        })

        print("fitting baseline reference...", flush=True)
        t0 = time.time()
        base_probs, _ = evaluate(baseline, df, feature_cols, y, fold_list)
        base_nll = per_match_nll(base_probs, y_val)
        base_loss = float(base_nll.mean())
        mlflow.log_metric("baseline_log_loss", base_loss)
        print(f"baseline log loss: {base_loss:.5f}  [{time.time() - t0:.0f}s]\n",
              flush=True)

        results = []
        best = None  # (log_loss, params, model, nll, probs)
        started = time.time()

        for i, params in enumerate(combos, 1):
            t0 = time.time()
            with mlflow.start_run(nested=True, run_name=f"{i:03d}"):
                probs, model = evaluate(params, df, feature_cols, y, fold_list)
                nll = per_match_nll(probs, y_val)
                loss = log_loss(y_val, probs, labels=[0, 1, 2])
                brier = brier_score_multiclass(y_val, probs, n_classes=3)

                # Positive => this config beats the baseline on these rows.
                diff = base_nll - nll
                diff_mean = float(diff.mean())
                diff_se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
                t_stat = diff_mean / diff_se if diff_se > 0 else 0.0

                mlflow.log_params({**params, "model": args.model,
                                   "mode": args.mode,
                                   "random_state": RANDOM_STATE})
                mlflow.log_metrics({
                    "log_loss": loss,
                    "brier_score": brier,
                    "improvement_vs_baseline": diff_mean,
                    "improvement_se": diff_se,
                    "improvement_t": t_stat,
                })

                results.append({**params, "log_loss": loss, "brier": brier,
                                "diff": diff_mean, "se": diff_se, "t": t_stat})
                if best is None or loss < best[0]:
                    best = (loss, params, model, nll, probs)

            flag = "*" if abs(t_stat) > SIGNIFICANCE_T else " "
            desc = " ".join(f"{k}={params[k]:<5}" for k in varied)
            print(
                f"{i:3d}/{len(combos)} {desc} "
                f"ll={loss:.5f} vs base {diff_mean:+.5f} (t={t_stat:+.2f}){flag} "
                f"[{time.time() - t0:.1f}s]",
                flush=True,
            )

        print(f"\nswept {len(combos)} points in {time.time() - started:.0f}s")

        res = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
        print("\ntop 10 by val log loss ('*' = differs from baseline beyond noise):")
        for _, r in res.head(10).iterrows():
            flag = "*" if abs(r["t"]) > SIGNIFICANCE_T else " "
            desc = " ".join(f"{k}={r[k]:<5}" for k in varied)
            print(f"  {r['log_loss']:.5f} {flag} {desc} "
                  f"diff={r['diff']:+.5f} +/-{1.96 * r['se']:.5f}")

        best_loss, best_params, best_model, best_nll, best_probs = best

        # The full surface, not just the winner. A grid this shape is worth
        # reading as a curve per axis: an optimum sitting on an edge means the
        # real one is outside the grid, which is how `reg` came to need `reg2`.
        print("\nmean log loss by axis (marginalising over the other):")
        for axis in varied:
            cells = res.groupby(axis)["log_loss"].mean().sort_index()
            print(f"  {axis:12s} " + "  ".join(
                f"{k}:{v:.5f}" for k, v in cells.items()))

        boot = ra.paired_bootstrap(base_nll, best_nll)
        print(f"\nbest: {dict((k, best_params[k]) for k in varied)}  "
              f"log loss {best_loss:.5f}")
        print(f"  paired bootstrap vs the baseline recipe: "
              f"{boot['improvement']:+.5f} "
              f"[{boot['ci_low']:+.5f}, {boot['ci_high']:+.5f}]  "
              f"P(better) = {boot['p_better']:.3f}")
        mlflow.log_params({f"best_{k}": best_params[k] for k in varied})
        mlflow.log_metrics({"best_log_loss": best_loss,
                            "best_improvement": boot["improvement"],
                            "best_ci_low": boot["ci_low"],
                            "best_ci_high": boot["ci_high"]})
        mlflow.xgboost.log_model(
            best_model, name=f"sweep_best_{args.model}_{args.grid}_{args.mode}")

        n_sig = int((res["t"].abs() > SIGNIFICANCE_T).sum())
        print(f"{n_sig}/{len(res)} grid point(s) differ from baseline beyond noise")

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        res_path = RESULTS_DIR / f"{args.model}_{args.grid}_{args.mode}.csv"
        res.to_csv(res_path, index=False)

        # Probabilities, not losses. FINDINGS_MARKET.md open item 5 and
        # FINDINGS_VENUE.md open item 4: caching predict_proba is what made the
        # by-outcome tables and the nine-arm blend replay free in later phases,
        # and caching losses instead cost a 17-minute re-fit.
        probs_path = RESULTS_DIR / f"{args.model}_{args.grid}_{args.mode}_probs.npz"
        np.savez_compressed(
            probs_path,
            fixture_id=df["fixture_id"].to_numpy()[val_all],
            y=y_val,
            baseline=base_probs,
            best=best_probs,
            best_params=np.array([f"{k}={best_params[k]}" for k in varied]),
        )
        print(f"per-point results -> {res_path}")
        print(f"validation probabilities -> {probs_path}")

        rank_agreement(res, varied, args.model, args.grid, args.mode)

        print(f"\nbest model logged to mlflow experiment '{spec['experiment']}' "
              f"under parent run {parent.info.run_id}")
        print("nothing written to any model's local path or S3 - served models untouched")

        if n_sig == 0:
            print("\nNOTE: no config separated from the baseline. That is a result, not a "
                  "failure - it means this grid does not contain a real improvement.")
        print("CAUTION: the winner's log loss is selected on the same val rows it is "
              "scored on, so it is optimistically biased - best-of-N is still best-of-N "
              "under rolling origin. Treat it as a ranking, and read the per-axis curves "
              "above for whether the optimum is interior.")


if __name__ == "__main__":
    main()
