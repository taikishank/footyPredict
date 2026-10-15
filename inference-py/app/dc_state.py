"""The Dixon-Coles fit app/model_btts.py's blend reads its Poisson half from,
rebuilt by replaying the archive rather than persisted and mutated - the same
choice app/ratings_state.py made for `atkdef`, and for the same reason: a
fit's state after fixture N depends on every fixture before it, and a
persisted/mutated copy can drift from the one ml/ fits without anything
noticing.

One difference from ratings_state.py decides this module's shape rather than
copying it outright: the replay itself is slow. atkdef is a single online
pass, measured at 0.4s for the whole archive; Dixon-Coles refits per league
on a cadence (ml/goals_lib.fit_league, warm-started), measured at ~29s for
the arm this serves (DC_HALF_LIFE/DC_CADENCE, build_goals_features.py's own
DIAGNOSTIC_ARM). ratings_state.py can afford to rebuild inline on whichever
request notices the state is stale; blocking a request thread for 29s cannot
be the same decision, so this module rebuilds on a dedicated background
thread instead and a request only ever reads whatever the last build left in
place.

Serving needs less than training did, too: ml/build_goals_features.py emits
a value *per fixture* because every row needs its own pre-match state for
leak-free training. A live request is always asking about a fixture in the
future, so only each league's *final* fitted LeagueParams - the one state
every upcoming fixture in that league shares until the next refit or
background rebuild - needs to be kept.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd

from app.config import DC_CADENCE, DC_HALF_LIFE, DC_REFRESH_SECONDS
from app.db import get_results_index
from build_goals_features import final_params as _replay_final_params
from goals_lib import LeagueParams, btts_prob

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_params: dict[int, LeagueParams] = {}
_loaded_at: float = 0.0
_n_results: int = 0
_latest_result: datetime | None = None
_refresh_thread: threading.Thread | None = None


def _fit_all_leagues(results: list[dict], as_of: "pd.Timestamp | None" = None) -> dict[int, LeagueParams]:
    """The LeagueParams per league that are correct as of `as_of` (defaults
    to now).

    Delegates to build_goals_features.final_params(), which shares its
    refit-window walk with replay_arm() - the function ml/'s own offline
    build and diagnostics use. An earlier version of this function
    re-derived the same day-window arithmetic independently and it drifted
    from replay_arm()'s by a few days near a refit boundary, which is
    exactly the training/serving divergence ml/FINDINGS_SCOPE.md warns
    against; sharing the walk instead of re-deriving it closes that off by
    construction. `as_of` is the other half of that equivalence - see
    final_params()'s own docstring for why "replay completed fixtures and
    stop" alone can serve a stale fit. `as_of` is an explicit parameter
    (rather than this function always reading the clock) so
    tests/test_dc_state.py can assert exact agreement with replay_arm() at
    a fixed historical moment.

    `starting_at` arrives tz-aware (UTC) from Postgres; the offline `date`
    column is naive UTC, and _walk_league()'s day-difference arithmetic
    needs both callers using the same representation.
    """
    if not results:
        return {}
    index = pd.DataFrame(results).rename(columns={"starting_at": "date"})
    index["date"] = pd.to_datetime(index["date"], utc=True).dt.tz_localize(None)
    if as_of is None:
        as_of = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return _replay_final_params(index, DC_HALF_LIFE, DC_CADENCE, as_of=as_of)


def load_state() -> None:
    """Read the archive and refit every league. Call once at startup; safe
    to repeat from the background thread."""
    global _params, _loaded_at, _n_results, _latest_result

    started = time.monotonic()
    results = get_results_index()
    params = _fit_all_leagues(results)

    with _lock:
        _params = params
        _loaded_at = time.monotonic()
        _n_results = len(results)
        _latest_result = results[-1]["starting_at"] if results else None

    logger.info(
        "Dixon-Coles replay: %d leagues from %d completed fixtures in %.1fs "
        "(latest result %s)",
        len(params), len(results), time.monotonic() - started, _latest_result,
    )


def is_loaded() -> bool:
    return bool(_params)


def state_summary() -> dict:
    return {
        "n_leagues": len(_params),
        "n_results": _n_results,
        "latest_result": _latest_result.isoformat() if _latest_result else None,
        "age_seconds": round(time.monotonic() - _loaded_at, 1) if _params else None,
    }


def _refresh_loop() -> None:
    """The initial load plus every rebuild after it, all off the request
    path and off the startup path - measured at ~76s on inference-py's own
    interpreter (not the ~29s the ml/ venv gave the same arm), too slow for
    either app/ratings_state.py's inline refresh-on-request (fine at 0.4s)
    or for lifespan to await before serving its first request.

    A fixture asked about before the first replay finishes gets the
    classifier alone (app/model_btts.predict_proba_blended's fallback) -
    the same degraded-not-blocked contract app/model_btts.py's own
    best-effort loading already sets for a missing artifact.
    """
    while True:
        try:
            load_state()
        except Exception:
            logger.exception("Dixon-Coles replay failed; "
                              "serving the previous state (or none)")
        if DC_REFRESH_SECONDS <= 0:
            return
        time.sleep(DC_REFRESH_SECONDS)


def start_background_refresh() -> None:
    """Call once at startup. Spawns the thread that performs the first
    replay and every one after it; returns immediately rather than waiting
    for that first replay, which is the whole point."""
    global _refresh_thread
    if _refresh_thread is not None:
        return
    _refresh_thread = threading.Thread(target=_refresh_loop, daemon=True)
    _refresh_thread.start()


def btts_probability(league_id: int, home_id: int, away_id: int) -> float | None:
    """P(both teams score) for one upcoming fixture, or None for a league
    this replay has never fit (no completed fixtures yet) - the caller's
    cue to fall back to the classifier alone, the same best-effort contract
    app/model_btts.py's own docstring sets for the artifact being missing."""
    params = _params.get(league_id)
    if params is None:
        return None
    lam, mu = params.rates(home_id, away_id)
    return float(btts_prob(np.array([lam]), np.array([mu]), params.rho)[0])
