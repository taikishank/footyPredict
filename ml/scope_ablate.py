"""Which way should the train/serve form-window skew be closed - and what does
it cost to have left it open?

`ml/build_features.py` averages a club's last MAX_WINDOW matches across season
boundaries; `inference-py/app/db.py:76` averages its last MAX_WINDOW matches
inside the current season. The served v2 model was fitted under the first and
is queried under the second. scope_lib.py has the archive counts; the short
version is that 20.4% of training rows carry prior-season matches, 7.4% of
them can never be served at all, and 88% of August has no served prediction.

This is not primarily a feature hypothesis - no arm here is expected to be a
gain. It is the correctness phase: the published log loss is measured on an
input distribution the service does not produce, and until these arms are on
the table there is no honest number to compare any later phase against.

Arms (scope_lib.RECIPES):

    xseason       the training recipe, unchanged. Every published number in
                  this project was measured here, so it is the baseline.
    xseason_age   the same window plus matches_into_season / stale_slots per
                  side. The cheapest possible fix: if the model only needs to
                  know its window is stale, no query changes at all.
    scoped        the serving recipe, exactly as db.py runs it - in-season
                  only, partial windows averaged down to MIN_WINDOW.
    scoped_full8  in-season only and no partial windows, refusing early-season
                  fixtures outright rather than averaging three matches.

Two tables, because the arms deliberately cover different fixtures and each
table is dishonest in a way the other is not:

  * PAIRED, on the intersection every arm emits. A valid paired bootstrap -
    same fixtures, same folds - and the only place a log-loss difference means
    what it usually means. It is also the table that flatters `scoped_full8`,
    because the intersection excludes precisely the fixtures it refuses.

  * PRICED, on every fixture any arm emits, with an arm's refusals scored at
    a no-information prediction: the class base rate from that fold's training
    slice. Refusing to predict is a real product outcome (the Upcoming tab
    shows nothing), and a mean taken only over what an arm chose to answer
    cannot see it. The base rate is an assumption, and it is the assumption
    that decides `scoped_full8`, so both tables are printed side by side
    rather than one being called the result.

Then a by-month table, which is where the effect lives: from a club's ninth
in-season match the two recipes are identical, so anything real here is an
August/September number that a full-archive mean dilutes by a factor of six.

Machinery is rating_ablate.py's, imported rather than restated - same folds,
same train_v2 hyperparameters, same paired bootstrap.

Run:
    ./inference-py/bin/python ml/build_scope_features.py       # ~4 min, once
    ./inference-py/bin/python ml/scope_ablate.py --mode split
    ./inference-py/bin/python ml/scope_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import scope_lib
import train_v2 as tv2
from build_scope_features import OUT_PATH
from feature_lib import FORM_FIELD_ORDER

CGR = ["home_club_goals_relative", "away_club_goals_relative"]
STALENESS_COLS = [f"{side}_{f}" for side in ("home", "away")
                  for f in scope_lib.STALENESS_FIELDS]
N_CLASSES = len(tv2.LABEL_MAP)


def arm_columns(name: str) -> list[str]:
    recipe = scope_lib.RECIPES[name]
    form = [f"{side}_form_{f}__{recipe.form_source}"
            for side in ("home", "away") for f in FORM_FIELD_ORDER]
    return form + CGR + (STALENESS_COLS if recipe.staleness else [])


def load() -> pd.DataFrame:
    df = pd.read_csv(OUT_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    print(f"{len(df)} fixtures in the scope table")
    for name in scope_lib.RECIPES:
        n = int(df[f"{name}_eligible"].sum())
        print(f"  {name:13s} {n:6d} eligible  ({n / len(df):6.1%})")
    both = df[[f"{n}_eligible" for n in scope_lib.RECIPES]].all(axis=1)
    print(f"  {'intersection':13s} {int(both.sum()):6d} eligible  "
          f"({both.mean():6.1%})  <- the PAIRED table's rows\n")
    return df


def base_rate_losses(y_train: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Per-fixture log loss of predicting the fold's training class rates.

    What a refusal costs in the PRICED table. Deliberately the weakest honest
    prediction available - a served fixture with no model behind it shows no
    probabilities at all, and pricing a refusal any *better* than "the league
    average outcome" would be inventing a fallback the service does not have.
    """
    priors = np.bincount(y_train, minlength=N_CLASSES) / len(y_train)
    return -np.log(np.clip(priors[y_val], 1e-15, 1.0))


def score_arm(df: pd.DataFrame, name: str, y: np.ndarray,
              plan: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """(per-fixture loss over every scored fixture, index of those fixtures).

    The fold boundaries are computed once on the whole table and then
    intersected with the arm's eligibility, so every arm trains and validates
    on the same *dates* while covering different rows within them. Computing
    folds per arm instead would give the arms different chronological cuts and
    make the comparison meaningless.

    Refusals are priced here rather than in the caller so an arm's losses come
    back aligned to `df`'s index, which is what lets both tables be assembled
    by selection afterwards.
    """
    cols = arm_columns(name)
    eligible = df[f"{name}_eligible"].to_numpy()
    losses, index = [], []
    for train_idx, val_idx in plan:
        tr = train_idx[eligible[train_idx]]
        va = val_idx[eligible[val_idx]]
        refused = val_idx[~eligible[val_idx]]
        losses.append(ra.fit_and_score(df, cols, y, tr, va))
        index.append(va)
        if len(refused):
            losses.append(base_rate_losses(y[tr], y[refused]))
            index.append(refused)
    order = np.argsort(np.concatenate(index))
    return np.concatenate(losses)[order], np.concatenate(index)[order]


def table(title: str, losses: dict[str, np.ndarray], note: str) -> None:
    base = losses[scope_lib.BASELINE_ARM]
    print(f"\n{title}  ({len(base)} fixtures)")
    print(f"{'arm':14s} {'cols':>5s} {'log loss':>9s} {'vs xseason':>11s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 74)
    for name in scope_lib.RECIPES:
        n_cols = len(arm_columns(name))
        if name == scope_lib.BASELINE_ARM:
            print(f"{name:14s} {n_cols:5d} {base.mean():9.5f} {'-':>11s} "
                  f"{'-':>20s} {'-':>10s}")
            continue
        stats = ra.paired_bootstrap(base, losses[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:14s} {n_cols:5d} {losses[name].mean():9.5f} "
              f"{stats['improvement']:+11.5f} {ci:>20s} {stats['p_better']:10.3f}")
    print(note)


def by_month(df: pd.DataFrame, priced: dict[str, np.ndarray],
             scored_index: np.ndarray) -> None:
    """Where the skew actually is. The two recipes are identical from a club's
    ninth in-season match, so a full-archive mean averages the effect over six
    months in which no arm can differ from any other."""
    month = df["date"].dt.month.to_numpy()[scored_index]
    print(f"\nPRICED log loss by calendar month\n{'month':>6s} {'n':>6s} " +
          " ".join(f"{name:>13s}" for name in scope_lib.RECIPES))
    print("-" * (14 + 14 * len(scope_lib.RECIPES)))
    for m in range(1, 13):
        mask = month == m
        if not mask.any():
            continue
        cells = " ".join(f"{priced[name][mask].mean():13.5f}" for name in scope_lib.RECIPES)
        print(f"{m:6d} {int(mask.sum()):6d} {cells}")
    print("\nAugust and September are the whole phase; October on, the arms are "
          "the same\nrecipe and any difference there is fold noise.")


def early_season(df: pd.DataFrame, priced: dict[str, np.ndarray],
                 scored_index: np.ndarray) -> None:
    """The distribution question on its own, with coverage held out of it.

    Restricted to August and September fixtures `scoped` can actually serve,
    so every arm below predicted every row with a real model and no cell is a
    priced refusal. This is the one comparison that isolates the second
    defect: when the service *does* answer in August, it answers from a
    3-to-7-match in-season mean, while the model behind it was fitted on
    8-match cross-season vectors and the offline number for those same
    fixtures was computed the same way.

    scoped_full8 is absent because it refuses ~every row here; its case is
    the PRICED table, not this one.
    """
    sub = df.iloc[scored_index]
    mask = (sub["date"].dt.month.isin((8, 9)) & sub["scoped_eligible"]).to_numpy()
    arms = [n for n in scope_lib.RECIPES if n != "scoped_full8"]
    print(f"\nEARLY SEASON - Aug/Sep fixtures scoped can serve ({int(mask.sum())} fixtures, "
          f"no priced refusals)")
    print(f"{'arm':14s} {'log loss':>9s} {'vs xseason':>11s} {'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 68)
    base = priced[scope_lib.BASELINE_ARM][mask]
    for name in arms:
        if name == scope_lib.BASELINE_ARM:
            print(f"{name:14s} {base.mean():9.5f} {'-':>11s} {'-':>20s} {'-':>10s}")
            continue
        stats = ra.paired_bootstrap(base, priced[name][mask])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:14s} {priced[name][mask].mean():9.5f} {stats['improvement']:+11.5f} "
              f"{ci:>20s} {stats['p_better']:10.3f}")
    print("\nNegative 'vs xseason' on the scoped arm is the cost of the served "
          "window itself,\nnet of coverage - the number the published Aug/Sep log "
          "loss does not contain.")


def coverage(df: pd.DataFrame, scored_index: np.ndarray) -> None:
    print("\nCoverage on the scored fixtures - what each arm refuses to predict")
    print(f"{'arm':14s} {'served':>8s} {'refused':>8s} {'Aug served':>12s} "
          f"{'Sep served':>12s}")
    print("-" * 58)
    sub = df.iloc[scored_index]
    month = sub["date"].dt.month.to_numpy()
    for name in scope_lib.RECIPES:
        el = sub[f"{name}_eligible"].to_numpy()
        aug, sep = month == 8, month == 9
        print(f"{name:14s} {el.mean():8.1%} {1 - el.mean():8.1%} "
              f"{el[aug].mean():12.1%} {el[sep].mean():12.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    cache = OUT_PATH.parent / f"scope_losses_{args.mode}.npz"
    df = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    print(f"mode={args.mode}, {len(plan)} fold(s), "
          f"{sum(len(v) for _, v in plan)} validation fixtures total\n")

    priced, scored_index = {}, None
    if cache.exists():
        stored = np.load(cache)
        priced = {n: stored[n] for n in stored.files if n != "scored_index"}
        scored_index = stored["scored_index"]
        print(f"reusing cached per-fixture losses from {cache.name}\n")

    for name in scope_lib.RECIPES:
        if name in priced:
            continue
        losses, index = score_arm(df, name, y, plan)
        # Every arm scores every validation fixture once the refusals are
        # priced, so the indices are identical across arms by construction -
        # asserted rather than assumed, since the two tables below both
        # index into df by this one array.
        if scored_index is None:
            scored_index = index
        else:
            assert np.array_equal(index, scored_index), f"{name} scored a different row set"
        priced[name] = losses
        print(f"  fitted {name:13s} ({len(arm_columns(name))} cols)  "
              f"priced log loss {losses.mean():.5f}")
        np.savez(cache, scored_index=scored_index, **priced)

    # The paired table is the priced one restricted to fixtures no arm had to
    # refuse - so it contains no base-rate cells at all, and the two tables
    # differ only in which rows they average, never in how a row was scored.
    all_eligible = df[[f"{n}_eligible" for n in scope_lib.RECIPES]].all(axis=1).to_numpy()
    mask = all_eligible[scored_index]
    table("PAIRED - fixtures every arm emits", {n: l[mask] for n, l in priced.items()},
          "\nA clean paired bootstrap, and the table that flatters scoped_full8: the "
          "rows\nit refuses are exactly the rows excluded here.")
    table("PRICED - every scored fixture, refusals at the fold's class base rate", priced,
          "\nRefusing to predict is priced, so an arm cannot buy accuracy with "
          "coverage.\nThe base rate is an assumption; it is the one that decides "
          "scoped_full8.")
    coverage(df, scored_index)
    early_season(df, priced, scored_index)
    by_month(df, priced, scored_index)


if __name__ == "__main__":
    main()
