"""A club's own rolling BTTS rate, pre-match - the one thing in v3's
63-column set that no column asks directly: not "how many goals has this
club scored/conceded" but "in how many of its last matches did *both* sides
score". atkdef's attack/defence ratings are fit on goals scored and conceded
separately; a club that is high-scoring and leaky and a club that is
high-scoring and miserly can carry the same two ratings while BTTS on
their matches differs a lot, and neither rate alone is the AND of the two.

Driven over the *full* results archive, not the 39,594-row join train_btts.py
trains on - a club's history should not be truncated by whichever inner join
the caller happens to need, the same reasoning form_lib.OpponentsFaced
documents for its own window.
"""
from __future__ import annotations

from collections import deque

WINDOW = 8


class BttsRate:
    """Per-club rolling mean of "both teams scored" over its last WINDOW
    matches, read strictly before record() for the match it is about to
    describe."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self._history: dict[int, deque[int]] = {}

    def rate(self, team_id: int) -> float | None:
        history = self._history.get(team_id)
        if not history:
            return None
        return sum(history) / len(history)

    def record(self, team_id: int, both_scored: int) -> None:
        history = self._history.get(team_id)
        if history is None:
            history = self._history.setdefault(team_id, deque(maxlen=self.window))
        history.append(both_scored)


def build(index) -> "pd.DataFrame":  # noqa: F821
    """(fixture_id, home_btts_rate, away_btts_rate) for every fixture in
    `index`, in date order. None (-> NaN) for a club's first `window`-or-fewer
    matches, same early-season gap build_features.py's MIN_WINDOW gate exists
    for - left as NaN rather than imputed, since the caller already drops
    rows without a full rating history and can drop these the same way.
    """
    import numpy as np
    import pandas as pd

    index = index.sort_values("date").reset_index(drop=True)
    tracker = BttsRate()
    home_rate = np.full(len(index), np.nan)
    away_rate = np.full(len(index), np.nan)

    for i, row in enumerate(index.itertuples()):
        home_id, away_id = int(row.home_id), int(row.away_id)
        h = tracker.rate(home_id)
        a = tracker.rate(away_id)
        if h is not None:
            home_rate[i] = h
        if a is not None:
            away_rate[i] = a
        both = int(row.home_goals > 0 and row.away_goals > 0)
        tracker.record(home_id, both)
        tracker.record(away_id, both)

    return pd.DataFrame({
        "fixture_id": index["fixture_id"].to_numpy(),
        "home_btts_rate": home_rate,
        "away_btts_rate": away_rate,
    })
