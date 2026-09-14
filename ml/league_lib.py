"""League identity and league character, as model columns.

Two different claims are on the table, and they need different columns.

*Identity* is "this fixture is in league 384". One column, or five dummies;
the model learns whatever it likes about each league, and learns nothing it
can apply to a league it has never seen. league_dummies() below.

*Character* is "this fixture is in a league that has lately been drawing 26%
of its matches and scoring 2.6 goals a game". Continuous, causal, and
transferable: a new league gets a sensible vector as soon as it has any
history, and two leagues that play alike share one. LeagueCharacter below.

The distinction matters because FINDINGS_LEAGUE.md's opening measurement is
that per-league *base rates* do not survive out of time - a league's own home
rate swings further across eras than the leagues differ from each other. A
rolling character column tracks that drift by construction; a dummy cannot.

Everything here is computed strictly from fixtures that kicked off earlier,
the same discipline ratings_lib and LeagueRollingMean follow: features(), then
update(), never the reverse.
"""
from __future__ import annotations

import math
from collections import deque

import pandas as pd

# One season of a 20-club league is 380 fixtures. Rolling rather than
# expanding for the reason FINDINGS_LEAGUE.md's drift table gives: the
# quantity being estimated moves, so an all-time mean is an era average and
# not a description of the league a fixture is actually being played in.
WINDOW = 380

# Below this the rate estimates are noisier than the between-league spread
# they are meant to capture (3-4pp; 100 fixtures puts a one-sigma band at
# 5pp), so the columns are NaN and xgboost routes those rows itself rather
# than reading a number that means nothing.
MIN_FIXTURES = 200


class LeagueCharacter:
    """A league's recent playing character, as of now.

    Four fields, each a rolling summary over the league's last WINDOW
    fixtures:

        league_home_rate    how often the home side wins
        league_draw_rate    how often nobody does
        league_goals        goals per fixture, both sides
        league_balance      the standard deviation of the goal margin - a
                            league where results are lopsided scores high,
                            one where everyone beats everyone narrowly scores
                            low. The cheapest available proxy for competitive
                            balance that needs no rating state.
    """

    FIELD_ORDER = (
        "league_home_rate",
        "league_draw_rate",
        "league_goals",
        "league_balance",
    )

    def __init__(self, window: int = WINDOW, min_fixtures: int = MIN_FIXTURES) -> None:
        self.window = window
        self.min_fixtures = min_fixtures
        self._recent: dict[int, deque[tuple[int, int]]] = {}

    def update(self, league_id: int, home_goals: int, away_goals: int) -> None:
        recent = self._recent.get(league_id)
        if recent is None:
            recent = self._recent[league_id] = deque(maxlen=self.window)
        recent.append((home_goals, away_goals))

    def features(self, league_id: int) -> dict[str, float]:
        recent = self._recent.get(league_id)
        if recent is None or len(recent) < self.min_fixtures:
            return {field: float("nan") for field in self.FIELD_ORDER}

        n = len(recent)
        margins = [h - a for h, a in recent]
        home = sum(1 for m in margins if m > 0) / n
        draw = sum(1 for m in margins if m == 0) / n
        goals = sum(h + a for h, a in recent) / n
        mean_margin = sum(margins) / n
        balance = math.sqrt(sum((m - mean_margin) ** 2 for m in margins) / n)
        return {
            "league_home_rate": home,
            "league_draw_rate": draw,
            "league_goals": goals,
            "league_balance": balance,
        }


def build_character(index: pd.DataFrame) -> pd.DataFrame:
    """One row of character columns per fixture in `index`, in date order.

    `index` is results_index.parquet: fixture_id, league_id, date, home_goals,
    away_goals. Each fixture is described by the league as it stood *before*
    that fixture kicked off, so the table can be joined onto the feature table
    without leaking a result into its own predictors.
    """
    index = index.sort_values("date")
    character = LeagueCharacter()
    rows = []
    for fx in index.itertuples():
        rows.append({"fixture_id": fx.fixture_id, **character.features(fx.league_id)})
        character.update(fx.league_id, fx.home_goals, fx.away_goals)
    return pd.DataFrame(rows)


def league_dummies(league_ids: pd.Series) -> pd.DataFrame:
    """One 0/1 column per league, named league_is_<id>.

    Columns are sorted by id so the set is stable whatever order the table
    arrives in, and a league absent from a training fold still gets its column
    - the alternative is an arm whose column count depends on the fold.
    """
    ids = sorted(league_ids.unique())
    return pd.DataFrame(
        {f"league_is_{i}": (league_ids == i).astype(int).to_numpy() for i in ids},
        index=league_ids.index,
    )
