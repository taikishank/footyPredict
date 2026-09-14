"""Does the model need to know which league a fixture is in?

The v3 recipe carries no league identity at all. Its form columns are club
averages, club_goals_relative divides out the league's scoring level on
purpose, and atkdef's ratings are league-scoped internally but expose only
per-club numbers. Nothing tells the trees whether they are looking at a
Premier League fixture or a Ligue 1 one - and the five leagues do not settle
their matches alike (FINDINGS_LEAGUE.md, headline table).

Five arms, ordered from the cheapest form of the claim to the most expensive:

    id          league_id as one numeric column. What most people reach for,
                and the weakest: xgboost can only split it into arbitrary
                groupings of unordered ids, so "8 and 564 behave alike" costs
                it several splits to express.

    onehot      one 0/1 column per league. The same information, reachable in
                a single split. id vs onehot is a pure encoding question and
                a disagreement between them is a fact about trees, not about
                football.

    character   four rolling, causal league descriptors (league_lib): recent
                home rate, draw rate, goals per fixture, and the spread of
                the goal margin. Not identity - two leagues that play alike
                share a vector, a league's own vector moves with its era, and
                a league the model has never seen still gets one. This is the
                arm that answers the drift result rather than ignoring it.

    interact    onehot plus its products with atkdef's expectations, so
                "a +0.4 expected goal difference means less here" is one
                split rather than a deep sequence of them. A strict superset
                of onehot, so the pairing of the two isolates slope from
                intercept.

    placebo     interact's shape with a random five-way partition of the
                fixtures in place of the league. Trees cannot express a
                product at max_depth=4 without spending a level on it, so an
                explicit product column can help for reasons that have
                nothing to do with what it multiplies. If this arm gains what
                interact gains, the finding is about column geometry and not
                about football.

    per_league  five separate models, one per league, each trained only on
                its own fixtures. The maximal version: every column
                interacted with league, at the cost of a fifth of the
                training data each. If league identity matters at all, this
                arm has the best chance of showing it and the worst data to
                show it with - and which of those wins is the finding.

The baseline is the served v3 recipe, so every arm is "v3 plus league", not
"form plus league". Everything else is rating_ablate.py's machinery, imported
rather than restated: the same expanding-window folds, the same paired
bootstrap on per-fixture losses, the same train_v2 hyperparameters.

The hyperparameters were tuned on the pooled model (tune.py:259), which is a
real caveat for per_league only: a model with a fifth of the rows probably
wants a shallower tree. A null there is "not at this tuning", and the arm is
cheap enough to re-run if the pooled arms make it worth the grid.

Run:
    ./inference-py/bin/python ml/league_ablate.py --mode split
    ./inference-py/bin/python ml/league_ablate.py --mode rolling
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import rating_ablate as ra
import train_v2 as tv2
from league_lib import LeagueCharacter, build_character, league_dummies
from ratings_lib import ARMS

FEATURES_DIR = Path(__file__).parent / "data" / "features"
INDEX_PATH = FEATURES_DIR / "results_index.parquet"

ATKDEF = list(ARMS["atkdef"].FIELD_ORDER)
CHARACTER = list(LeagueCharacter.FIELD_ORDER)

# The expectations, not the four level columns: atkdef_d exists in
# rating_ablate for the reason build_rating_features.py --diagnose gives (the
# levels still correlate +0.17 with the date), and an interaction is only
# worth forming with a column that means the same thing in every era.
INTERACT_WITH = ["exp_goals_home", "exp_goals_away", "exp_goal_diff"]

BASELINE = "v3"

PER_LEAGUE = "per_league"

# The placebo's partition is drawn once, from fixture_id, so it is stable
# across runs and across folds - a partition redrawn per fold would be a
# different feature in every model and would fail for that reason rather than
# for the reason the control exists to test.
PLACEBO_GROUPS = 5
PLACEBO_SEED = 0


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
    """The v3 table plus every league column any arm needs.

    Inner joins throughout, so a fixture missing from any input is missing
    from all arms - the fixture-for-fixture pairing the bootstrap relies on.
    """
    df, baseline = ra.load()
    baseline = baseline + ATKDEF

    character = build_character(pd.read_parquet(INDEX_PATH))
    before = len(df)
    df = df.merge(character, on="fixture_id", how="inner").sort_values("date")
    df = df.reset_index(drop=True)
    assert len(df) == before, f"{before - len(df)} fixtures lack character columns"

    dummies = league_dummies(df["league_id"])
    dummy_cols = list(dummies.columns)

    rng = np.random.default_rng(PLACEBO_SEED)
    fake = pd.Series(rng.integers(0, PLACEBO_GROUPS, len(df)), index=df.index)
    placebo = league_dummies(fake).rename(columns=lambda c: c.replace("league_is_", "fake_is_"))
    placebo_cols = list(placebo.columns)

    df = pd.concat([df, dummies, placebo], axis=1)
    for dummy in dummy_cols + placebo_cols:
        for signal in INTERACT_WITH:
            df[f"{dummy}_x_{signal}"] = df[dummy] * df[signal]

    missing = df[CHARACTER[0]].isna().mean()
    print(f"{len(df)} fixtures, {len(baseline)} baseline columns, "
          f"{missing:.1%} without league character (a league's first "
          f"{LeagueCharacter().min_fixtures} fixtures)")
    return df, baseline, dummy_cols, placebo_cols


def _interacted(dummies: list[str]) -> list[str]:
    return dummies + [f"{d}_x_{s}" for d in dummies for s in INTERACT_WITH]


def arm_columns(name: str, baseline: list[str], dummies: list[str],
                placebo: list[str] | None = None) -> list[str]:
    if name == BASELINE or name == PER_LEAGUE:
        return baseline
    if name == "id":
        return baseline + ["league_id"]
    if name == "onehot":
        return baseline + dummies
    if name == "character":
        return baseline + CHARACTER
    if name == "interact":
        return baseline + _interacted(dummies)
    if name == "placebo":
        return baseline + _interacted(placebo or [])
    raise KeyError(name)


ARMS_ORDER = [BASELINE, "id", "onehot", "character", "interact", "placebo",
              PER_LEAGUE]


def fit_and_score_per_league(df: pd.DataFrame, cols: list[str], y: np.ndarray,
                             train_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
    """One model per league, scored back into val_idx's own order.

    The order matters more than it looks: every other arm returns losses in
    val_idx order, and the bootstrap pairs arms by position. Scattering into a
    pre-allocated array rather than concatenating per-league blocks is what
    keeps this arm comparable with the rest.
    """
    league = df["league_id"].to_numpy()
    losses = np.full(len(val_idx), np.nan)
    for value in np.unique(league[val_idx]):
        train = train_idx[league[train_idx] == value]
        seats = np.flatnonzero(league[val_idx] == value)
        if len(train) == 0:
            # A league that appears in a fold's validation slice with no
            # earlier fixtures of its own. Impossible on the current archive
            # (every league starts before the first cut), and a silent zero
            # or a crash inside xgboost would both be worse than saying so.
            raise ValueError(
                f"league {value} has {len(seats)} validation fixtures and no "
                "training rows in this fold"
            )
        losses[seats] = ra.fit_and_score(df, cols, y, train, val_idx[seats])
    assert not np.isnan(losses).any(), "a validation fixture went unscored"
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("split", "rolling"), default="split")
    args = parser.parse_args()

    cache = FEATURES_DIR / f"league_losses_{args.mode}.npz"
    df, baseline, dummies, placebo = load()
    y = df["result"].map(tv2.LABEL_MAP).to_numpy()
    plan = ra.folds(len(df), args.mode)
    print(f"mode={args.mode}, {len(plan)} fold(s), "
          f"{sum(len(v) for _, v in plan)} validation fixtures total\n")

    results = {}
    if cache.exists():
        stored = np.load(cache)
        results = {name: stored[name] for name in stored.files}
        print(f"reusing cached per-fixture losses from {cache.name}\n")

    for name in ARMS_ORDER:
        if name in results:
            continue
        cols = arm_columns(name, baseline, dummies, placebo)
        fit = fit_and_score_per_league if name == PER_LEAGUE else ra.fit_and_score
        losses = np.concatenate([fit(df, cols, y, tr, va) for tr, va in plan])
        results[name] = losses
        label = f"{len(cols)} cols" + (" x 5 models" if name == PER_LEAGUE else "")
        print(f"  fitted {name:12s} ({label})  log loss {losses.mean():.5f}")
        np.savez(cache, **results)

    base = results[BASELINE]
    print(f"\n{'arm':12s} {'cols':>5s} {'log loss':>9s} {'vs v3':>9s} "
          f"{'95% CI':>20s} {'P(better)':>10s}")
    print("-" * 70)
    for name in ARMS_ORDER:
        cols = arm_columns(name, baseline, dummies, placebo)
        if name == BASELINE:
            print(f"{name:12s} {len(cols):5d} {base.mean():9.5f} "
                  f"{'-':>9s} {'-':>20s} {'-':>10s}")
            continue
        stats = ra.paired_bootstrap(base, results[name])
        ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
        print(f"{name:12s} {len(cols):5d} {results[name].mean():9.5f} "
              f"{stats['improvement']:+9.5f} {ci:>20s} {stats['p_better']:10.3f}")

    print("\nPositive 'vs v3' favours the arm. An interval spanning zero is a "
          "null result\nregardless of where the point estimate sits.")

    # Intercept against slope. Both arms carry the same five dummies, so this
    # pairing sees only the products - the question the phase exists to ask,
    # since FINDINGS_LEAGUE.md's opening measurement already killed the
    # intercept on its own.
    stats = ra.paired_bootstrap(results["onehot"], results["interact"])
    print(f"\ninteracting league with the rating expectations is worth "
          f"{stats['improvement']:+.5f} over the\nsame dummies alone "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]")

    # The control. Same column count, same shape, no league in it.
    stats = ra.paired_bootstrap(results["placebo"], results["interact"])
    print(f"\ninteract over its random-partition placebo: "
          f"{stats['improvement']:+.5f} "
          f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]\n"
          "A null here means the products help for a reason that is not the "
          "league.")

    # Where each arm's gain (or loss) actually landed. A pooled null hiding
    # one league that genuinely wants its own model, and four that are
    # damaged by the split, is a different result from a flat nothing.
    league = df["league_id"].to_numpy()[np.concatenate([v for _, v in plan])]
    print(f"\nvs v3, per league:\n\n{'arm':12s}" +
          "".join(f"{value:>10d}" for value in np.unique(league)))
    print("-" * (12 + 10 * len(np.unique(league))))
    for name in ARMS_ORDER[1:]:
        row = "".join(
            f"{(base[league == value] - results[name][league == value]).mean():>+10.5f}"
            for value in np.unique(league)
        )
        print(f"{name:12s}{row}")


if __name__ == "__main__":
    main()
