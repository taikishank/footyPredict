"""Where does the challenger's gain actually come from?

`train_challenger.py` answers "is the block worth shipping". This answers the
follow-up questions that one raises and cannot settle:

  blocks    each tier on its own and together, at one row set - separates
            "this block is redundant" from "this block is weak"
  features  within the availability block, experience against everything else
  trim      the performance block cut down to its top-k features by importance,
            testing whether 36 columns on 14,403 training rows is simply too
            many for the signal to survive
  weighting counted absence against minutes-weighted absence, and flat
            per-position means against form-weighted ones - the rework of the
            aggregation that Phase 9's findings pointed at
  isolate   each block *without* team form, against the base-rate floor - the
            scale on which "+0.006" should be read, and whether the squad
            information exists at all or only fails to be additional
  where     which fixtures the squad block actually helps on, since a feature
            about disrupted line-ups should earn its keep on disrupted line-ups
            and be inert on the other 80%
  involve   the three importance measures side by side - counting heads,
            weighting by minutes, weighting by goal involvement - plus the
            collinearity diagnostic that explains the result

Every arm is trained on identical rows and an identical split and compared with
the same paired bootstrap as the promotion bar itself, so the numbers here are
directly comparable with `train_challenger.py`'s.

    ./inference-py/bin/python ml/player_stats/ablate.py blocks --tier performance
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))

import train_challenger as tc  # noqa: E402
from player_feature_lib import (  # noqa: E402
    ABSENCE_FIELD_ORDER,
    AVAILABILITY_BASE_FIELDS,
    INVOLVEMENT_FIELD_ORDER,
    INVOLVEMENT_SHARE_FIELDS,
    INVOLVEMENT_VOLUME_FIELDS,
    AVAILABILITY_FIELD_ORDER,
    PERFORMANCE_BASE_FIELDS,
    PERFORMANCE_FIELD_ORDER,
    WEIGHTED_AVAILABILITY_FIELDS,
    WEIGHTED_PERFORMANCE_FIELDS,
    WEIGHTED_SQUAD_FIELDS,
)


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


def weighting(tier: str) -> None:
    """Does weighting players by how much they play beat treating them alike?

    Phase 9's finding was that `key_players_absent` ranks a club's eleventh-most
    used player level with its top scorer, and that the only performance feature
    that ranked at all was a distribution statistic. Both point at the
    aggregation, so this asks the question directly and in both tiers:

      counted absence     vs  minutes-weighted absence      (availability, 2006+)
      flat position means vs  + form-weighted squad quality  (performance, 2016+)

    The absence arms *swap* rather than stack, because the whole claim is that
    the weighted column says something the counted one cannot; if it only helps
    alongside the count, that is a different and much weaker result.
    """
    df, form, sides, y, split = load(tier)
    counted = [n for n in AVAILABILITY_BASE_FIELDS]
    swapped = [n for n in counted if n != "key_players_absent"] + list(
        WEIGHTED_AVAILABILITY_FIELDS
    )

    arms = {
        "form only": form,
        "counted absence": form + sides(counted),
        "weighted absence": form + sides(swapped),
        "both absence forms": form + sides(AVAILABILITY_FIELD_ORDER),
    }
    pairs = [
        ("form only", "counted absence"),
        ("form only", "weighted absence"),
        ("counted absence", "both absence forms"),
        ("weighted absence", "both absence forms"),
    ]

    if any(f"home_sq_{n}" in df.columns for n in WEIGHTED_PERFORMANCE_FIELDS):
        availability = sides(AVAILABILITY_FIELD_ORDER)
        arms["+ flat performance"] = form + availability + sides(PERFORMANCE_BASE_FIELDS)
        arms["+ weighted squad"] = (
            form + availability + sides(PERFORMANCE_BASE_FIELDS + WEIGHTED_SQUAD_FIELDS)
        )
        arms["+ absent quality"] = (
            form + availability + sides(PERFORMANCE_BASE_FIELDS + ABSENCE_FIELD_ORDER)
        )
        arms["+ all weighted"] = form + availability + sides(PERFORMANCE_FIELD_ORDER)
        pairs += [
            ("both absence forms", "+ flat performance"),
            ("+ flat performance", "+ weighted squad"),
            ("+ flat performance", "+ absent quality"),
            ("+ flat performance", "+ all weighted"),
            ("both absence forms", "+ all weighted"),
        ]

    results = {name: fit(df, cols, y, split) for name, cols in arms.items()}
    report(results, pairs)
    _print_resolution(y, split)


def involve(tier: str) -> None:
    """Goal involvement as the importance measure, against the other two.

    `weighting` established that minutes-weighted absence is +0.981 correlated
    with counted absence, because the key eleven are *selected* by minutes -
    asking the same question twice. Goal involvements are not how those players
    were picked and they concentrate far more steeply, so this is the measure
    with variation independent of the team sheet.

    Three absence arms, each *replacing* `key_players_absent` rather than
    stacking on it, so each is a like-for-like swap of the importance measure:

      counted      how many of the key eleven are missing
      minutes      what share of the first team's playing time is missing
      involvement  what share of the club's recent scoring is missing

    The diagnostic at the end is half the point. If involvement absence is as
    collinear with the count as minutes absence was, the null result needs no
    further explanation; if it is not, then a genuinely independent measure has
    been tried and the effect is simply absent.
    """
    df, form, sides, y, split = load(tier)
    base = [n for n in AVAILABILITY_BASE_FIELDS]
    without_count = [n for n in base if n != "key_players_absent"]

    # The share arms are the swap test the findings asked for. The volume arms
    # are here because the first run of this experiment conflated the two and
    # credited the absence weighting with a gain that belonged entirely to club
    # scoring volume - which is not a player feature at all. Separating them is
    # the lesson, so the separation is now part of the experiment.
    arms = {
        "form only": form,
        "counted": form + sides(base),
        "minutes-weighted": form + sides(without_count + list(WEIGHTED_AVAILABILITY_FIELDS)),
        "involvement shares": form + sides(without_count + list(INVOLVEMENT_SHARE_FIELDS)),
        "+ club goals only": form + sides(base) + sides(["club_goals_relative"]),
        "+ involvement volume": form + sides(base) + sides(INVOLVEMENT_VOLUME_FIELDS),
        "shares + volume": form + sides(without_count + list(INVOLVEMENT_FIELD_ORDER)),
        "everything": form + sides(AVAILABILITY_FIELD_ORDER),
    }
    results = {name: fit(df, cols, y, split) for name, cols in arms.items()}
    report(results, [
        ("form only", "counted"),
        ("counted", "minutes-weighted"),
        ("counted", "involvement shares"),
        ("counted", "+ club goals only"),
        ("counted", "+ involvement volume"),
        ("+ club goals only", "+ involvement volume"),
        ("+ involvement volume", "shares + volume"),
        ("counted", "everything"),
    ])
    _print_resolution(y, split)
    _involvement_diagnostic(df)


def _involvement_diagnostic(df: pd.DataFrame) -> None:
    """How much of each absence measure is not just the count restated.

    A feature that correlates +0.98 with the column it replaces cannot carry
    new information no matter how well-motivated it is, and that is a property
    of the data worth printing next to the log loss rather than inferring from
    it.
    """
    count = "home_sq_key_players_absent"
    print("\nhow independent is each importance measure of the count?")
    # The date column is not decoration: this project shipped an era proxy by
    # accident once already, and the raw volume counts correlate +0.405 with the
    # date before normalisation.
    print(f"{'feature':30s} {'corr':>7} {'std':>8} {'residual std':>13} {'vs date':>8}")
    for name in ("key_minutes_absent", "involvement_absent", "squad_involvement_absent",
                 "xi_involvement_share", "top_player_absent", "top_scorer_absent",
                 "involvement_volume_relative", "club_goals_relative"):
        col = f"home_sq_{name}"
        if col not in df.columns:
            continue
        r = df[col].corr(df[count])
        std = df[col].std()
        date_r = df[col].corr(df["date"].astype("int64"))
        print(f"  {name:28s} {r:+.3f} {std:8.4f} {std * (1 - r**2) ** 0.5:13.4f} {date_r:+8.3f}")


def _base_rate_floor(y, split) -> dict:
    """What the training set's class priors alone score on the validation rows.

    This is the zero-information benchmark, and the reason it is printed first:
    a log loss reduction is meaningless until you know how much reducible loss
    there was. Football outcomes are mostly irreducible, so the whole distance
    between priors and the best model we have is small - and a feature worth a
    tenth of that distance is not a small feature, whatever the third decimal
    place looks like.
    """
    priors = np.bincount(y[:split], minlength=3) / len(y[:split])
    probs = np.tile(priors, (len(y) - split, 1))
    return {
        "log_loss": log_loss(y[split:], probs, labels=[0, 1, 2]),
        "losses": tc.per_row_log_loss(y[split:], probs),
        "cols": [],
    }


def isolate(tier: str) -> None:
    """Each block alone, with no team form at all.

    `blocks` and `involve` both ask whether a block adds anything *on top of*
    team form, which cannot distinguish "this block knows nothing" from "team
    form already knows it". Team form is eight matches of results produced by
    these same players, so the second is the live hypothesis and this is the
    mode that tests it: strip team form out and see how far the squad columns
    get on their own.
    """
    df, form, sides, y, split = load(tier)
    availability = sides(AVAILABILITY_FIELD_ORDER)
    performance = sides(PERFORMANCE_FIELD_ORDER)
    squad = availability + performance

    arms = {"base rates only": None, "form only": form, "availability only": availability}
    if performance:
        arms["performance only"] = performance
    arms["squad only"] = squad
    arms["form + squad"] = form + squad

    results = {}
    for name, cols in arms.items():
        results[name] = _base_rate_floor(y, split) if cols is None else fit(df, cols, y, split)

    report(results, [(a, b) for a, b in zip(list(arms), list(arms)[1:])])

    floor = results["base rates only"]["log_loss"]
    print(f"\nreducible loss, priors -> best arm here: "
          f"{floor - min(r['log_loss'] for r in results.values()):.5f}")
    print("each arm as a share of that total reducible loss:")
    for name, r in results.items():
        print(f"  {name:22s} {floor - r['log_loss']:+.5f}  "
              f"{(floor - r['log_loss']) / (floor - min(x['log_loss'] for x in results.values())):6.1%}")

    _redundancy(df, form, squad, split)


def _redundancy(df, form, squad, split) -> None:
    """How much of each squad column team form can already predict.

    Least-squares fit of each squad column on the 54 team-form columns, on the
    training rows only. A column that team form explains at R^2 = 0.9 cannot
    carry much that is new, however well-motivated it is as football.
    """
    X = df[form].iloc[:split].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), np.nan_to_num(X)])
    scores = {}
    for col in squad:
        target = df[col].iloc[:split].to_numpy(dtype=float)
        keep = ~np.isnan(target)
        if keep.sum() < 100:
            continue
        beta, *_ = np.linalg.lstsq(X[keep], target[keep], rcond=None)
        resid = target[keep] - X[keep] @ beta
        var = target[keep].var()
        scores[col] = 1 - resid.var() / var if var > 0 else float("nan")

    ranked = sorted(scores, key=lambda c: -scores[c])
    print(f"\nhow much of each squad column team form already explains (R^2, train rows):")
    print(f"  mean {np.nanmean(list(scores.values())):.3f}   "
          f"median {np.nanmedian(list(scores.values())):.3f}")
    print("  most redundant:")
    for col in ranked[:5]:
        print(f"    {scores[col]:.3f}  {col}")
    print("  least redundant:")
    for col in ranked[-5:]:
        print(f"    {scores[col]:.3f}  {col}")


def where(tier: str) -> None:
    """Which fixtures the squad block's gain comes from.

    The theory behind the block is specific: knowing the XI should matter when
    the XI is unusual. If that is right, the gain should concentrate on heavily
    rotated or depleted line-ups and be ~zero on the settled ones - in which
    case an average over all fixtures understates the effect where it applies,
    and the block is worth more than the headline suggests for exactly the
    fixtures a punter cares about. If the gain is flat across rotation levels,
    the block is a squad-quality proxy and the spec's framing is wrong.
    """
    df, form, sides, y, split = load(tier)
    squad = sides(AVAILABILITY_FIELD_ORDER) + sides(PERFORMANCE_FIELD_ORDER)
    baseline = fit(df, form, y, split)
    challenger = fit(df, form + squad, y, split)
    delta = baseline["losses"] - challenger["losses"]  # positive favours squad

    val = df.iloc[split:].reset_index(drop=True)
    print(f"overall gain {delta.mean():+.5f} over {len(delta):,} fixtures\n")

    absent = val["home_sq_key_players_absent"] + val["away_sq_key_players_absent"]
    groups = {
        "settled (0-3 of 22 out)": absent <= 3,
        "some rotation (4-6)": (absent >= 4) & (absent <= 6),
        "heavy rotation (7-9)": (absent >= 7) & (absent <= 9),
        "wholesale (10+)": absent >= 10,
    }
    if "home_sq_top_scorer_absent" in val.columns:
        scorers = val["home_sq_top_scorer_absent"] + val["away_sq_top_scorer_absent"]
        groups["neither top scorer out"] = scorers == 0
        groups["one top scorer out"] = scorers == 1
        groups["both top scorers out"] = scorers == 2

    rng = np.random.default_rng(tc.RNG_SEED)
    print(f"{'subset':28s} {'n':>6} {'gain':>9}  {'95% CI':>22}")
    for name, mask in groups.items():
        sel = delta[mask.to_numpy()]
        if len(sel) < 50:
            print(f"  {name:26s} {len(sel):>6}   too few")
            continue
        boot = sel[rng.integers(0, len(sel), size=(2000, len(sel)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  {name:26s} {len(sel):>6} {sel.mean():+9.5f}  [{lo:+.5f}, {hi:+.5f}]")


MODES = {
    "blocks": blocks,
    "isolate": isolate,
    "where": where,
    "features": features,
    "trim": trim,
    "weighting": weighting,
    "involve": involve,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--tier", choices=tc.TIERS, default="performance")
    args = parser.parse_args()
    MODES[args.mode](args.tier)


if __name__ == "__main__":
    main()
