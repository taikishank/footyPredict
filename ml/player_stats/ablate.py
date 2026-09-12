"""Where does the challenger's gain actually come from?

`train_challenger.py` answers "is the block worth shipping". This answers the
follow-up questions that one raises and cannot settle:

  blocks    each tier on its own and together, at one row set - separates
            "this block is redundant" from "this block is weak"
  features  within the availability block, experience against everything else
  trim      the performance block cut down to its top-k features by importance,
            testing whether 36 columns on 14,403 training rows is simply too
            many for the signal to survive

Every arm is trained on identical rows and an identical split and compared with
the same paired bootstrap as the promotion bar itself, so the numbers here are
directly comparable with `train_challenger.py`'s.

    ./inference-py/bin/python ml/player_stats/ablate.py blocks --tier performance
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))

import train_challenger as tc  # noqa: E402
from player_feature_lib import AVAILABILITY_FIELD_ORDER, PERFORMANCE_FIELD_ORDER  # noqa: E402


def load(tier: str):
    df = pd.read_csv(tc.TIERS[tier], parse_dates=["date"]).sort_values("date")
    form = [c for c in df.columns if c.startswith(("home_form_", "away_form_"))]

    def sides(names):
        # home and away always move together: a feature that helps one side
        # helps the other, and splitting a pair would just make the model
        # learn an asymmetry that is not in the game.
        return [f"{s}_sq_{n}" for s in ("home", "away") for n in names if f"{s}_sq_{n}" in df.columns]

    y = df["result"].map(tc.LABEL_MAP).to_numpy()
    split = int(len(df) * (1 - tc.VAL_FRACTION))
    return df, form, sides, y, split


def fit(df, cols, y, split):
    X = df[cols]
    probs, model = tc.fit_and_score(X.iloc[:split], y[:split], X.iloc[split:], y[split:])
    return {
        "log_loss": log_loss(y[split:], probs, labels=[0, 1, 2]),
        "losses": tc.per_row_log_loss(y[split:], probs),
        "model": model,
        "cols": cols,
    }


def report(results: dict, pairs: list[tuple[str, str]]) -> None:
    for name, r in results.items():
        print(f"  {name:24s} {len(r['cols']):>4} cols   {r['log_loss']:.5f}")
    print()
    for a, b in pairs:
        test = tc.paired_bootstrap(results[a]["losses"], results[b]["losses"])
        verdict = (
            "clears" if test["ci_low"] > 0 else "HURTS" if test["ci_high"] < 0 else "noise"
        )
        print(f"  {a:22s} -> {b:22s} {test['improvement']:+.5f} "
              f"CI [{test['ci_low']:+.5f}, {test['ci_high']:+.5f}]  {verdict}")


def blocks(tier: str) -> None:
    """Each tier alone and together. Weak and redundant look identical in a
    nested comparison; only running the performance block *without* the
    availability one tells them apart."""
    df, form, sides, y, split = load(tier)
    arms = {
        "form only": form,
        "form + availability": form + sides(AVAILABILITY_FIELD_ORDER),
        "form + performance": form + sides(PERFORMANCE_FIELD_ORDER),
        "form + both": form + sides(AVAILABILITY_FIELD_ORDER) + sides(PERFORMANCE_FIELD_ORDER),
    }
    arms = {name: cols for name, cols in arms.items() if len(cols) > len(form) or name == "form only"}
    results = {name: fit(df, cols, y, split) for name, cols in arms.items()}
    report(results, [
        ("form only", "form + availability"),
        ("form only", "form + performance"),
        ("form + performance", "form + both"),
        ("form + availability", "form + both"),
    ])
    _print_resolution(y, split)


def features(tier: str) -> None:
    """Experience against the rest of the availability block."""
    df, form, sides, y, split = load(tier)
    rest = [n for n in AVAILABILITY_FIELD_ORDER if n != "xi_relative_experience"]
    arms = {
        "form only": form,
        "+ experience": form + sides(["xi_relative_experience"]),
        "+ team news": form + sides(rest),
        "+ all availability": form + sides(AVAILABILITY_FIELD_ORDER),
    }
    results = {name: fit(df, cols, y, split) for name, cols in arms.items()}
    report(results, [
        ("form only", "+ experience"),
        ("form only", "+ team news"),
        ("+ experience", "+ all availability"),
        ("+ team news", "+ all availability"),
    ])
    _print_resolution(y, split)


def trim(tier: str) -> None:
    """The performance block cut to its top-k features by importance.

    The ranking comes from a model fit on the training rows only, so it is not
    itself val-set shopping - but picking the best k off the table below is,
    and a k that wins by less than the resolution printed at the end has not
    been shown to win at all.
    """
    df, form, sides, y, split = load(tier)
    base_cols = form + sides(AVAILABILITY_FIELD_ORDER)
    full = fit(df, base_cols + sides(PERFORMANCE_FIELD_ORDER), y, split)

    importance = pd.Series(full["model"].feature_importances_, index=full["cols"])
    paired = {
        name: importance.get(f"home_sq_{name}", 0) + importance.get(f"away_sq_{name}", 0)
        for name in PERFORMANCE_FIELD_ORDER
    }
    ranked = sorted(paired, key=lambda n: -paired[n])
    print("performance features by paired importance:")
    for name in ranked:
        print(f"  {paired[name]:.4f}  {name}")

    base = fit(df, base_cols, y, split)
    print(f"\nform + availability, no performance block: {base['log_loss']:.5f}\n")
    print(f"{'top-k':>6} {'cols':>5} {'log loss':>9}   vs form + availability")
    for k in (1, 2, 3, 4, 6, 8, 12, len(ranked)):
        arm = fit(df, base_cols + sides(ranked[:k]), y, split)
        test = tc.paired_bootstrap(base["losses"], arm["losses"])
        verdict = (
            "clears" if test["ci_low"] > 0 else "HURTS" if test["ci_high"] < 0 else "noise"
        )
        print(f"{k:>6} {len(arm['cols']):>5} {arm['log_loss']:>9.5f}   "
              f"{test['improvement']:+.5f} "
              f"CI [{test['ci_low']:+.5f}, {test['ci_high']:+.5f}]  {verdict}")
    _print_resolution(y, split)


def _print_resolution(y, split) -> None:
    """The floor under every 'noise' verdict above.

    A validation set this size cannot resolve an arbitrarily small effect, and
    reporting "no difference" without saying how small a difference would have
    been invisible overstates the finding.
    """
    print(f"\n{len(y) - split:,} validation fixtures; this split resolves effects of "
          f"roughly 0.005 log loss and larger.")


MODES = {"blocks": blocks, "features": features, "trim": trim}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--tier", choices=tc.TIERS, default="performance")
    args = parser.parse_args()
    MODES[args.mode](args.tier)


if __name__ == "__main__":
    main()
