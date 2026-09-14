"""Opponent context for the form window.

feature_lib.team_form() averages a club's last MAX_WINDOW matches without
recording who those matches were against. This is the smallest thing that
fixes that: the mean attack and defence rating of the opponents the *same*
window saw, so a form vector can be read against the schedule that produced
it.

Deliberately separate from feature_lib.py. Nothing here is on the serving
path yet - ml/FINDINGS_OPPONENT.md's pre-registration has to clear first -
and feature_lib is the module training and inference share, so an
experimental column does not belong in it.

The window length is the caller's, defaulting to the served one, for the
reason build_window_features.py gives: the opponents-faced mean has to cover
exactly the matches team_form() averages, or it describes a different
schedule from the one in the form columns.
"""
from __future__ import annotations

from collections import deque

from feature_lib import MAX_WINDOW

#: Exact key order faced() returns, same contract as FORM_FIELD_ORDER.
FACED_FIELD_ORDER = ("opp_atk_faced", "opp_def_faced")


class OpponentsFaced:
    """Per-club rolling mean of the opponent ratings the form window saw.

    Two fields rather than one scalar because the mechanism is directional:
    an opponent's *defence* suppresses the club's attacking output (shots,
    possession, corners) while its *attack* inflates what the club concedes,
    and a single "strength faced" column would blur the two together on the
    27 fields where they pull in opposite directions.

    Ratings are the opponent's *pre-match* values, on AttackDefence's log
    scale, so higher `def` means a better defence (the update subtracts it)
    and higher `atk` a better attack.

    Driven like Rating: faced() strictly before record() for each fixture, so
    a row never sees the opponent it is about to play.
    """

    def __init__(self, window_size: int = MAX_WINDOW) -> None:
        self.window_size = window_size
        self._history: dict[int, deque[tuple[float, float]]] = {}

    def faced(self, team_id: int) -> dict[str, float] | None:
        """The window's mean opponent ratings, or None for a club with no
        history at all - build_features.py's MIN_WINDOW gate means a club in
        an emitted row always has some."""
        history = self._history.get(team_id)
        if not history:
            return None
        n = len(history)
        return {
            "opp_atk_faced": sum(atk for atk, _ in history) / n,
            "opp_def_faced": sum(dfn for _, dfn in history) / n,
        }

    def record(self, team_id: int, opp_atk: float, opp_def: float) -> None:
        history = self._history.get(team_id)
        if history is None:
            history = self._history.setdefault(team_id, deque(maxlen=self.window_size))
        history.append((opp_atk, opp_def))
