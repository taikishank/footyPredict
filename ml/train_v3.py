"""Trains v3: the v2 feature set plus the `atkdef` rating columns, fitted with
the capacity the retune actually asked for rather than the one v2 handed down.

v3's columns are settled and were settled elsewhere - FINDINGS_RATINGS.md picked
`atkdef` over `elo`, `gdelo` and the union of all three, and this script does not
re-open that. What is new here is the recipe. Every v3 number this project has
published was measured at ml/train_v2.py's hyperparameters, which is the right
default for an ablation (it measures the columns, not the fit) and the wrong one
for an artifact: those constants were selected by ml/tune.py --grid reg/reg2 on
v2's 56 columns, before the seven rating columns existed.

Retuning them changed the answer, and changed a published finding with it.

    ml/tune.py --grid cap     extends toward MORE capacity, testing the reading
                              of two placebos (FINDINGS_GOALS.md open item 2,
                              FINDINGS_LEAGUE.md open item 1) that said v3 could
                              not reach the interactions its columns imply.
                              Depth came back monotone the other way, 3 through
                              8: 0.98390 0.98535 0.98706 0.98877 0.99217. v3 is
                              over-fitting at the served depth 4, not under it.
    ml/tune.py --grid cap2    extends downward, because `cap` won on its edge.
                              max_depth 1-3 x reg_lambda 10-100 x 800/2400
                              rounds; the top ten configs land within 0.00073.
    ml/seed_probe.py          prices the seed band at 0.00007 SD, which is what
                              makes a 0.00012 gap between the leaders unreadable
                              and a 0.00032 placebo real.
    ml/cap_runoff.py          pairs the leaders against each other on identical
                              fixtures, and re-runs the placebo at each depth.

The runoff settled it twice over. All four candidates beat the served recipe by
+0.0017 to +0.0020 and no pair of them separates - every pairwise interval spans
zero - so log loss cannot choose between them and the tie is broken on grounds
that are not log loss. And the placebo gain died monotonically with depth:

    served  depth 4   +0.00032  [+0.00003, +0.00062]
    d3      depth 3   +0.00004  [-0.00017, +0.00024]
    d2      depth 2   -0.00004  [-0.00014, +0.00007]
    d1      depth 1   -0.00017  [-0.00033, +0.00000]

which is the over-capacity reading and not the under-capacity one: noise columns
helped because they absorbed splits real columns would otherwise have wasted,
an accidental colsample_bytree. Fix the depth and the effect is gone. Both open
items close with the sign opposite to the one they guessed.

MAX_DEPTH = 2 is therefore chosen over the nominally-better depth-1 config on
the same argument train_v2.py's SUBSAMPLE comment makes for 0.6 over 0.5: it is
interior on the axis rather than sitting on the grid's edge, it needs 800 rounds
where depth 1 needs 2400 to buy the same capacity, and the placebo is already
flat there. Being nominally 0.00012 better is not a reason when the seed band is
0.00007 and the two configs do not separate.

Two ways this script deliberately differs from train_v2.py:

1. **Publishing is opt-in.** train_v2.py uploads to S3 as a side effect of
   training, so every experimental fit ships. Here the upload is behind
   --publish and nothing else touches the bucket.
2. **Both protocols are reported.** The artifact is fitted and scored on the
   single chronological holdout, as v2's is, but the recipe above was selected
   under rolling origin and the split is the protocol this project has twice
   caught disagreeing with it. Printing both means the artifact's headline
   number is never the one the pick was not made on.

Open, and not decided here: like v2's, the shipped artifact is fitted on the
first 80% of the archive and never sees the most recent fifth. Refitting on
everything before serving is the obvious thing to want and would make v2 and v3
no longer comparable on the shipped weights, so it is a decision for the serving
path (FINDINGS_CAPACITY.md), not a change to sneak in with a retune.

Run:
    ./inference-py/bin/python ml/train_v3.py
    ./inference-py/bin/python ml/train_v3.py --publish
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow  # noqa: E402
import mlflow.xgboost  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

import rating_ablate as ra  # noqa: E402
import train_v2 as tv2  # noqa: E402

MODEL_LOCAL_PATH = Path(__file__).parent / "models" / "match_outcome_model_v3.json"

# Same bucket as the incumbent and v2 (ml/train.py, ml/train_v2.py); a third key
# so all three artifacts coexist and the serving path chooses between them.
S3_BUCKET_MODELS = os.environ.get("S3_BUCKET_MODELS", "liveedge-model-artifacts-590184129653")
S3_KEY = "2000_v3_atkdef/model.json"

MLFLOW_TRACKING_DIR = Path(__file__).parent / "mlruns"
EXPERIMENT_NAME = "liveedge-match-outcome-v3"  # ml/tune.py --model v3 lands here too

# The rating arm, decided in FINDINGS_RATINGS.md and not re-opened here.
RATING_ARM = "atkdef"

# Carried over from v2 unchanged. LEARNING_RATE was swept by `core` (36 points,
# nothing beat it beyond noise) and held fixed through `cap` and `cap2`, so it
# has never been swept *at depth 2* - see the open item in FINDINGS_CAPACITY.md.
# SUBSAMPLE likewise: `reg2` found its curve flat 0.5-0.7 on v2's columns and
# `cap`/`cap2` both held it at 0.6 rather than triple a grid.
LEARNING_RATE = tv2.LEARNING_RATE
SUBSAMPLE = tv2.SUBSAMPLE
VAL_FRACTION = tv2.VAL_FRACTION
LABEL_MAP = tv2.LABEL_MAP

# The retune's answer. See the docstring for why depth 2 and not the nominally
# better depth 1, and why 800 rounds rather than 2400.
N_ESTIMATORS = 800
MAX_DEPTH = 2
REG_LAMBDA = 30

# Pinned for reproducibility, as v2's is. With SUBSAMPLE < 1 this is not inert:
# ml/seed_probe.py measures 0.00007 SD of pooled log loss across seeds, which is
# the resolution limit on every number this recipe produces.
RANDOM_STATE = 0

#: The single place the retuned constants live, in the shape XGBClassifier and
#: ml/tune.py's evaluate() both take. Ablations re-priced at v3's capacity should
#: import this rather than restate it, for the reason rating_ablate.fit_and_score
#: gives: a restated constant that silently drifts is a confound that reads as a
#: finding. Note the direction of the dependency - this module imports
#: rating_ablate, so rating_ablate must never import it back.
PARAMS = {
    "n_estimators": N_ESTIMATORS,
    "max_depth": MAX_DEPTH,
    "learning_rate": LEARNING_RATE,
    "reg_lambda": REG_LAMBDA,
    "subsample": SUBSAMPLE,
    "random_state": RANDOM_STATE,
}


#: The fits an ablation may be run at, by name. Lives here rather than in
#: rating_ablate because only this module can see both: rating_ablate owns the
#: v2 recipe and must not import this one (the dependency runs the other way).
#:
#: `v2` is every published finding's fit and stays every ablation's default, so
#: a plain run still reproduces its FINDINGS file. `v2` is not obsolete and this
#: is not a migration - it is the recipe the whole board was measured at, and a
#: re-priced arm is only interpretable next to the number it replaces.
RECIPES = {"v2": ra.V2_PARAMS, "v3": PARAMS}


def build() -> XGBClassifier:
    """An unfitted v3 classifier. One constructor, so a re-priced ablation and
    the shipped artifact cannot drift apart."""
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        **PARAMS,
    )


def load() -> tuple:
    """The v3 table: rating_ablate's join, restricted to the chosen arm.

    Going through rating_ablate.load() rather than reading and joining the CSVs
    here is what keeps the artifact's rows identical to the 39,594 fixtures
    every v3 finding is measured on.
    """
    df, base_cols = ra.load()
    cols = base_cols + ra.ARM_COLUMNS[RATING_ARM]
    y = df["result"].map(LABEL_MAP).to_numpy()
    return df, cols, y


def brier_score_multiclass(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((one_hot - probs) ** 2, axis=1)))


def score(df, cols, y, fold_list) -> tuple[float, float, XGBClassifier]:
    """Pooled log loss and Brier over every fold's validation slice.

    Returns the last fold's model, which under `split` is the only fit there was
    and under `rolling` is the one trained on the most history.
    """
    parts, model = [], None
    for train_idx, val_idx in fold_list:
        model = build()
        model.fit(df[cols].iloc[train_idx], y[train_idx])
        parts.append(model.predict_proba(df[cols].iloc[val_idx]))
    probs = np.vstack(parts)
    val_all = np.concatenate([val for _, val in fold_list])
    y_val = y[val_all]
    return (log_loss(y_val, probs, labels=[0, 1, 2]),
            brier_score_multiclass(y_val, probs, n_classes=3),
            model)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--publish", action="store_true",
                    help="upload the artifact to S3. Off by default: training a "
                         "model and shipping it are separate decisions.")
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df, cols, y = load()
    split_folds = ra.folds(len(df), "split")
    rolling_folds = ra.folds(len(df), "rolling")

    (train_idx, val_idx), = split_folds
    print(f"{len(cols)} features ({len(cols) - len(ra.ARM_COLUMNS[RATING_ARM])} "
          f"from v2 + {len(ra.ARM_COLUMNS[RATING_ARM])} {RATING_ARM})")
    print(f"train: {len(train_idx)} rows, val: {len(val_idx)} rows "
          f"(chronological split)")
    print(f"recipe: {PARAMS}\n")

    with mlflow.start_run():
        mlflow.log_params({
            **PARAMS,
            "rating_arm": RATING_ARM,
            "val_fraction": VAL_FRACTION,
            "n_features": len(cols),
            "n_rows": len(df),
            "n_train_rows": len(train_idx),
            "n_val_rows": len(val_idx),
        })

        # The artifact's own holdout first: this fit is the one that gets saved.
        loss, brier, model = score(df, cols, y, split_folds)
        print(f"split    log loss {loss:.5f}   brier {brier:.5f}")
        mlflow.log_metrics({"log_loss": loss, "brier_score": brier})

        # And the protocol the recipe was actually selected under. Scored, not
        # saved: the artifact should be the fit whose reported number is its own
        # holdout, and pooling five folds' probabilities gives a loss that
        # belongs to no single model.
        r_loss, r_brier, _ = score(df, cols, y, rolling_folds)
        print(f"rolling  log loss {r_loss:.5f}   brier {r_brier:.5f}  "
              f"({len(rolling_folds)} folds; the protocol the recipe was picked on)")
        mlflow.log_metrics({"rolling_log_loss": r_loss,
                            "rolling_brier_score": r_brier})

        mlflow.xgboost.log_model(model, name="v3_atkdef")

        MODEL_LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODEL_LOCAL_PATH))
        print(f"\nsaved model artifact to {MODEL_LOCAL_PATH}")

        if args.publish:
            import boto3
            boto3.client("s3").upload_file(
                str(MODEL_LOCAL_PATH), S3_BUCKET_MODELS, S3_KEY)
            mlflow.log_param("s3_key", S3_KEY)
            print(f"uploaded model artifact to s3://{S3_BUCKET_MODELS}/{S3_KEY}")
        else:
            print("not published (pass --publish to upload to "
                  f"s3://{S3_BUCKET_MODELS}/{S3_KEY})")

        print(f"run logged to mlflow experiment '{EXPERIMENT_NAME}' at {MLFLOW_TRACKING_DIR}")


if __name__ == "__main__":
    main()
