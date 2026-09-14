"""How much of the gap to the market has the model actually closed?

Every ablation this project has run measures an arm against the *previous
arm*. That answers "is this change worth shipping" and cannot answer "is
there anything left worth chasing" - a +0.0076 gain reads as enormous against
a +0.0044 regularization sweep and as rounding error against a model that
trails the closing line by 0.03. FINDINGS_RATINGS.md sets the reducible
budget with class priors (0.0876), but a prior is a floor, not a target:
nobody knows whether a feature can reach it. The market is the target,
because a bookmaker's closing line is the best-known public forecast of the
same event.

The benchmark set is Pinnacle's *closing* prices from football-data.co.uk,
de-vigged - see build_market_index.py for why that source, and market_lib for
how the margin is removed. Closing rather than opening because closing is the
price after team news, which is the number a model actually has to beat.

Scored on the same rows and in the same order as rating_ablate.py --mode
rolling, and against its cached per-fixture losses, so the model figures here
are the *identical* numbers in FINDINGS_RATINGS.md rather than a re-fit that
might differ. That cache is required: this script deliberately fits nothing.

The comparison is a paired bootstrap on per-fixture losses, matching the
promotion bar in PLAYER_STATS_SPEC.md section 6 - model and market price the
same fixtures, so resampling the difference lets a chaotic week inflate both
losses together and cancel.

Run:
    ./inference-py/bin/python ml/rating_ablate.py --mode rolling   # once, ~17 min
    ./inference-py/bin/python ml/market_benchmark.py
    ./inference-py/bin/python ml/market_benchmark.py --source pinnacle_open
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import train_v2 as tv2
from rating_ablate import ARM_COLUMNS, N_FOLDS, RNG_SEED, load, paired_bootstrap

FEATURES_DIR = Path(__file__).parent / "data" / "features"
MARKET_PATH = FEATURES_DIR / "market.csv"
CACHE_PATH = FEATURES_DIR / "rating_losses_rolling.npz"

# Matches market_lib.DEVIG_METHODS. Both are reported because the two make
# different assumptions about where a bookmaker puts its margin, and the
# benchmark should not rest on that choice being right - if they agree, the
# number can be quoted without a caveat.
DEVIG_METHODS = ("shin", "multiplicative")

OUTCOME_ORDER = ("home", "draw", "away")  # train_v2.LABEL_MAP's 0, 1, 2


def validation_fixture_ids(df: pd.DataFrame) -> np.ndarray:
    """The fixture_ids rating_ablate --mode rolling scores, in its own order.

    Reconstructed rather than stored, so the two cannot drift apart silently:
    the folds are contiguous expanding-window slices covering the back half of
    the (date-sorted) table, and the cached loss arrays are those slices
    concatenated. So the pooled losses line up element-for-element with the
    back half of df, and any change to the fold plan breaks the length
    assertion below rather than quietly misaligning the pairing.
    """
    cuts = np.linspace(len(df) // 2, len(df), N_FOLDS + 1).astype(int)
    return df["fixture_id"].to_numpy()[cuts[0]:cuts[-1]]


def market_losses(market: pd.DataFrame, source: str, method: str,
                  y: np.ndarray) -> np.ndarray:
    columns = [f"{source}_{method}_{outcome}" for outcome in OUTCOME_ORDER]
    probs = market[columns].to_numpy()
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-15, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="pinnacle_closing",
                        help="price column set from build_market_index.PRICE_SETS")
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        raise SystemExit(
            f"{CACHE_PATH.name} not found - run "
            "`ml/rating_ablate.py --mode rolling` first. This script fits nothing."
        )

    df, _ = load()
    y_all = df["result"].map(tv2.LABEL_MAP).to_numpy()
    val_ids = validation_fixture_ids(df)

    cached = np.load(CACHE_PATH)
    model_losses = {name: cached[name] for name in cached.files}
    for name, losses in model_losses.items():
        assert len(losses) == len(val_ids), (
            f"{name} has {len(losses)} losses against {len(val_ids)} validation "
            "fixtures - the fold plan in rating_ablate.py has changed since the "
            "cache was written; delete the cache and re-run it."
        )

    # Align the market onto the validation rows by position, so every array
    # below is indexed the same way as the cached losses.
    market = pd.read_csv(MARKET_PATH)
    market = market.set_index("fixture_id").reindex(val_ids)
    y_val = y_all[len(df) // 2:]

    priced = market[f"{args.source}_{DEVIG_METHODS[0]}_home"].notna().to_numpy()
    print(f"\n{priced.sum()} of {len(val_ids)} validation fixtures have "
          f"{args.source} prices ({priced.mean() * 100:.1f}%)")
    if not priced.any():
        raise SystemExit(f"no {args.source} prices on the validation rows")

    # Every number below is computed on the priced subset only, model included
    # - a model figure over all 19,797 rows and a market figure over 18,907 of
    # them would not be comparable, and the difference between them would be
    # partly a difference in which fixtures were scored.
    market = market[priced]
    y_priced = y_val[priced]
    model_losses = {name: losses[priced] for name, losses in model_losses.items()}

    prior = np.bincount(y_priced, minlength=3) / len(y_priced)
    prior_loss = -np.log(prior[y_priced]).mean()

    print(f"\n{'forecaster':34s} {'log loss':>9s}")
    print("-" * 45)
    print(f"{'class priors (no information)':34s} {prior_loss:9.5f}")
    for name in ["baseline (v2)"] + list(ARM_COLUMNS):
        print(f"{'model: ' + name:34s} {model_losses[name].mean():9.5f}")
    market_by_method = {}
    for method in DEVIG_METHODS:
        losses = market_losses(market, args.source, method, y_priced)
        market_by_method[method] = losses
        print(f"{'market: ' + args.source + ' / ' + method:34s} "
              f"{losses.mean():9.5f}")

    spread = abs(market_by_method["shin"].mean()
                 - market_by_method["multiplicative"].mean())
    print(f"\nthe two de-vig methods differ by {spread:.5f}, so the de-vig "
          f"choice is\n{'not ' if spread < 0.0005 else ''}load-bearing here.")

    reference = market_by_method["shin"]
    print(f"\nEach model against the market ({args.source}, shin de-vigged). "
          f"Positive favours\nthe model, so every row here is expected to be "
          f"negative - the question is by how much.\n")
    print(f"{'model':20s} {'vs market':>10s} {'95% CI':>22s} {'P(model better)':>16s}")
    print("-" * 72)
    for name in ["baseline (v2)"] + list(ARM_COLUMNS):
        stats = paired_bootstrap(reference, model_losses[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:20s} {stats['improvement']:+10.5f} {ci:>22s} "
              f"{stats['p_better']:16.3f}")

    # The framing FINDINGS_RATINGS.md uses, with the market in place of the
    # class-prior floor: the prior is what a forecaster with no information
    # scores, the market is the best public forecast of the same fixtures, and
    # the gap between them is the budget anything is competing for.
    budget = prior_loss - reference.mean()
    print(f"\nThe market beats no-information by {budget:.5f}. Against that as "
          f"100%:\n")
    print(f"{'model':20s} {'log loss':>9s} {'share of the market gap closed':>32s}")
    print("-" * 64)
    for name in ["baseline (v2)"] + list(ARM_COLUMNS):
        closed = (prior_loss - model_losses[name].mean()) / budget
        print(f"{name:20s} {model_losses[name].mean():9.5f} {closed * 100:31.1f}%")
    print(f"\n{'market':20s} {reference.mean():9.5f} {100.0:31.1f}%")

    best = min(ARM_COLUMNS, key=lambda n: model_losses[n].mean())
    remaining = model_losses[best].mean() - reference.mean()
    print(f"\nThe best model arm ({best}) trails the closing line by "
          f"{remaining:.5f}.\nThat is the headroom any further feature work is "
          f"competing for - compare it\nwith the +0.0076 the entire ratings "
          f"phase bought.")

    breakdown(market, model_losses[best], reference, y_priced, best, args.source)


def breakdown(market: pd.DataFrame, model: np.ndarray, reference: np.ndarray,
              y: np.ndarray, arm: str, source: str) -> None:
    """Where the remaining gap actually sits.

    A single headroom figure says how much is left but not what to chase. The
    two cuts below are the ones that change what to build next: if the deficit
    is flat across every slice it is a general modelling gap, and if it
    concentrates it names the thing to work on.
    """
    gap = model - reference

    print(f"\n\nWhere the {arm} gap sits, by actual outcome:\n")
    print(f"{'outcome':10s} {'n':>6s} {'model':>9s} {'market':>9s} "
          f"{'gap':>9s} {'share of total gap':>20s}")
    print("-" * 68)
    total = gap.sum()
    for label, code in tv2.LABEL_MAP.items():
        mask = y == code
        print(f"{label:10s} {mask.sum():6d} {model[mask].mean():9.5f} "
              f"{reference[mask].mean():9.5f} {gap[mask].mean():+9.5f} "
              f"{gap[mask].sum() / total * 100:19.1f}%")

    # Bucketed on how one-sided the *market* thinks the fixture is, since that
    # is a pre-match quantity - bucketing on the model's own confidence would
    # sort partly on the model's errors.
    favourite = market[[f"{source}_shin_{o}" for o in OUTCOME_ORDER]].to_numpy().max(axis=1)
    edges = [0.0, 0.40, 0.50, 0.60, 0.75, 1.0]
    print(f"\nBy how one-sided the market thinks the fixture is:\n")
    print(f"{'market favourite':18s} {'n':>6s} {'model':>9s} {'market':>9s} "
          f"{'gap':>9s} {'share of total gap':>20s}")
    print("-" * 76)
    for lo, hi in zip(edges, edges[1:]):
        mask = (favourite >= lo) & (favourite < hi)
        if not mask.any():
            continue
        print(f"{f'{lo:.2f} - {hi:.2f}':18s} {mask.sum():6d} "
              f"{model[mask].mean():9.5f} {reference[mask].mean():9.5f} "
              f"{gap[mask].mean():+9.5f} {gap[mask].sum() / total * 100:19.1f}%")


if __name__ == "__main__":
    main()
