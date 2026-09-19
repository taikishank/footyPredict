"""The `atkdef` rating state v3 predicts from, rebuilt by replaying the
archive rather than persisted and mutated.

A rating is not a query. `team_form` reads a club's last eight matches and is
therefore a function of a bounded window, which is why app/features.py can
build it per request straight from Postgres. ratings_lib.AttackDefence's state
after fixture N depends on every fixture before it, back to 2000 - there is no
window to read.

So the state has to come from somewhere, and there were two candidates.

**A rating state store**: persist atk/dfn/mu/hfa, have the ingestor advance
them as results land. Buys O(1) startup. Costs a second implementation of a
numerically delicate update rule living next to the first, dual-write
consistency on every ingest, and - the part that decides it - drift that
nothing can detect. A rating that has silently diverged from the one the model
was fitted on produces confident, wrong probabilities, and no health check
short of replaying the archive would notice.

**Replay the archive on load**: one read of ~40k completed fixtures and a walk
through ratings_lib.Replay, the identical driver ml/build_rating_features.py
trains from. Correct by construction, no new failure mode, no new store.

That is a real tradeoff only if the replay is expensive, so it was measured
before it was chosen: **0.4 seconds** for the whole 26-season archive, one arm.
The store would have bought a fraction of a second and cost a class of bug this
project has already been bitten by once (ml/FINDINGS_SCOPE.md's form window).

Two properties this file depends on, both established rather than assumed:

- **Replay order is (starting_at, fixture_id).** 68% of fixtures share a
  kickoff instant, and the offline build used to break those ties with an
  unstable sort - deterministic given its parquet, and underivable from
  Postgres. See build_rating_features.order(); ordering here must match it
  exactly or the served ratings are not the trained ones.
- **The replayed state is never mutated.** roll_to() regresses ratings at a
  season boundary, so serving a fixture mutates whatever it is called on. Each
  request therefore rolls a private copy (~0.5 ms), which also makes the
  singleton safe to read from FastAPI's threadpool without a lock.
"""
from __future__ import annotations

import copy
import logging
import threading
import time
from datetime import datetime

from app.config import RATINGS_REFRESH_SECONDS
from app.db import get_results_index
from ratings_lib import ARMS, Replay

logger = logging.getLogger(__name__)

#: The promoted arm (ml/FINDINGS_RATINGS.md picked `atkdef` over `elo`,
#: `gdelo` and the union). Only this one is replayed - the offline build walks
#: all three so the arms sit on byte-identical rows, a comparison the service
#: has no use for and would pay 3x to reproduce.
ARM = "atkdef"

#: Column order v3's feature vector expects, which is the arm's own
#: FIELD_ORDER - the same contract ml/rating_ablate.ARM_COLUMNS reads.
FIELD_ORDER = ARMS[ARM].FIELD_ORDER

_lock = threading.Lock()
_replay: Replay | None = None
_loaded_at: float = 0.0
_n_results: int = 0
_latest_result: datetime | None = None


def load_state() -> None:
    """Read the archive and replay it. Call once at startup; safe to repeat."""
    global _replay, _loaded_at, _n_results, _latest_result

    started = time.monotonic()
    results = get_results_index()

    replay = Replay({ARM: ARMS[ARM]()})
    for row in results:
        # Same three calls in the same order as build_rating_features.build().
        # features() is skipped: the offline walk needs a pre-match row for
        # every fixture, and here only the final state matters.
        replay.roll_to(row["league_id"], row["home_id"], row["away_id"], row["starting_at"])
        replay.record(row["league_id"], row["home_id"], row["away_id"],
                      row["starting_at"], row["home_goals"], row["away_goals"])

    with _lock:
        _replay = replay
        _loaded_at = time.monotonic()
        _n_results = len(results)
        _latest_result = results[-1]["starting_at"] if results else None

    logger.info(
        "replayed %d completed fixtures in %.2fs (latest result %s)",
        len(results), time.monotonic() - started, _latest_result,
    )


def is_loaded() -> bool:
    return _replay is not None


def state_summary() -> dict:
    """What the state is, for /health. `n_results` and `latest_result` are the
    two numbers that say whether the served ratings are the trained ones: a
    count far from the archive's own means the service is reading a different
    history than ml/ did."""
    return {
        "arm": ARM,
        "n_results": _n_results,
        "latest_result": _latest_result.isoformat() if _latest_result else None,
        "age_seconds": round(time.monotonic() - _loaded_at, 1) if _replay else None,
    }


def _refresh_if_stale() -> None:
    """Rebuild when results have landed since the last replay.

    The form columns are read from Postgres per request and so are always
    current; without this the rating columns alone would age, and a model fed
    fresh form beside month-old ratings is being asked to predict from a row
    it was never fitted on. Time-based rather than event-based because the
    service is not told when the ingestor writes.
    """
    if _replay is None or RATINGS_REFRESH_SECONDS <= 0:
        return
    if time.monotonic() - _loaded_at < RATINGS_REFRESH_SECONDS:
        return
    load_state()


def rating_features(league_id: int, home_id: int, away_id: int,
                    kickoff: datetime) -> dict[str, float]:
    """The arm's pre-match columns for one upcoming fixture.

    Rolls a *copy* forward to the fixture's season, so an August fixture asked
    about in July sees the same post-rollover ratings the training replay gave
    it, and the singleton is left untouched for the next caller.
    """
    if _replay is None:
        raise RuntimeError("rating state not loaded - call load_state() first")
    _refresh_if_stale()

    local = copy.deepcopy(_replay)
    local.roll_to(league_id, home_id, away_id, kickoff)
    return local.features(home_id, away_id, league_id)
