"""Which model actually serves, resolved in one place.

main.py used to name `model_v2` directly at eight call sites, which made the
serving model a property of the request handlers rather than of configuration.
Adding v3 that way would have meant eight branches; this is the indirection
instead, so promoting a model is a SERVED_MODEL_VERSION change and nothing
else. The three loaders remain independent singletons - the point of that
design (see app/model_v2.py) is that a challenger cannot disturb the path
currently serving, and routing through one function does not change it.

v3 is the first model whose input is not purely a query over Postgres: its
seven rating columns are replayed state (app/ratings_state.py). That is why
load() is here too - "the model is loaded" and "the model can be fed" are
different claims for v3, and a health check that only knew the first would
report ok on a service that raises on every prediction.
"""
from __future__ import annotations

from app import model, model_v2, model_v3, ratings_state
from app.config import SERVED_MODEL_VERSION
from app.features import MatchFeatures

#: version tag -> (loader module, the MatchFeatures attribute it eats).
_SERVERS = {
    "v1": (model, "vector"),
    "v2": (model_v2, "v2_vector"),
    "v3": (model_v3, "v3_vector"),
}

if SERVED_MODEL_VERSION not in _SERVERS:
    raise RuntimeError(
        f"SERVED_MODEL_VERSION={SERVED_MODEL_VERSION!r} is not one of {sorted(_SERVERS)}"
    )

#: The loader module serving predictions, and the MatchFeatures attribute it
#: eats. Public because "which module actually serves" is this module's whole
#: answer, and a test that stubs the serving path has to stub the same module
#: the app will call - hardcoding v2 there is how tests/test_main.py went stale
#: at the v2 -> v3 promotion.
MODULE, VECTOR_ATTR = _SERVERS[SERVED_MODEL_VERSION]

_module, _vector_attr = MODULE, VECTOR_ATTR

#: v3 needs its rating state replayed before it can be fed. Kept as a property
#: of the version rather than a check inside load(), so a future model that
#: also needs state declares it here rather than editing control flow.
NEEDS_RATINGS = SERVED_MODEL_VERSION == "v3"


def load() -> None:
    """Load whatever the served model needs. A missing piece is fatal, for the
    reason the old lifespan comment gave: starting up without it would either
    serve nothing or silently fall back to a model nobody is grading."""
    _module.load_model()
    if NEEDS_RATINGS:
        ratings_state.load_state()
    # Nothing is loaded for a model that is not serving. Replaying v3's state
    # while v2 serves would let the challenger be scored in the shadow of the
    # incumbent, which is worth having - but it is a second archive read and a
    # second failure mode at startup, so it belongs to whatever asks for shadow
    # scoring rather than to every boot.


def is_loaded() -> bool:
    return _module.is_loaded() and (not NEEDS_RATINGS or ratings_state.is_loaded())


def model_version() -> str:
    return _module.model_version()


def predict_proba(features: MatchFeatures) -> dict[str, float]:
    vector = getattr(features, _vector_attr)
    if vector is None:
        # Only reachable for a model whose input is more than a Postgres query,
        # i.e. v3 without rating state. Explicit, because the alternative is a
        # numpy error three frames down that reads like a model bug.
        raise RuntimeError(
            f"no {_vector_attr} for fixture {features.fixture['fixture_id']} - "
            f"{SERVED_MODEL_VERSION} cannot be served without it"
        )
    return _module.predict_proba(vector)
