if __name__ == "__main__":
    # Allow `python3 app/main.py` / `python3 inference-py/app/main.py` to work
    # regardless of cwd - running this file directly only puts its own
    # directory (app/) on sys.path, not inference-py/, so `import app` fails.
    # `uvicorn app.main:app` from inference-py/ doesn't need this.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import db, model_btts, model_comparison, ratings_state, served
from app.config import (
    CORS_ALLOWED_ORIGINS,
    EDGE_FLAG_THRESHOLD,
    SERVED_BTTS_MODEL_VERSION,
    SERVED_MODEL_VERSION,
)
from app.features import NotEnoughHistoryError, build_match_features
from app.live_prob import adjust_for_live_state
from app.schemas import (
    BTTSProbabilities,
    HealthResponse,
    ModelComparisonResponse,
    ModelPerformance,
    LivePredictionResponse,
    PredictionResponse,
    Probabilities,
    UpcomingFixture,
    UpcomingFixturesResponse,
)

# How often the WS stream re-reads live match state (DynamoDB) and re-sends
# if changed. ingestor-go's live poller writes it on its own interval (see
# ingestor-go/internal/live/service.go); this just needs to be frequent
# enough that dashboard updates feel live without hammering DynamoDB.
_LIVE_STREAM_POLL_SECONDS = 4

# Matches the dashboard's Upcoming tab window (PROJECT_SPEC.md Phase 4).
_UPCOMING_WINDOW = timedelta(days=3)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open()
    # Whichever model config.SERVED_MODEL_VERSION names, plus whatever that
    # model needs to be fed - for v3 that includes replaying the rating state.
    # A missing piece is fatal rather than something to degrade past: starting
    # up without it would either serve nothing or silently fall back to a model
    # nobody is grading. See app/served.py.
    served.load()
    # Best-effort, not fatal: see app/model_btts.py's docstring for why BTTS
    # is a second, optional prediction rather than a candidate for
    # SERVED_MODEL_VERSION's slot.
    model_btts.load_model()
    yield
    db.pool.close()


app = FastAPI(title="LiveEdge Inference", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if served.is_loaded() else "model not loaded",
        model_version=served.model_version() if served.is_loaded() else None,
        ratings=ratings_state.state_summary() if ratings_state.is_loaded() else None,
        btts_model_loaded=model_btts.is_loaded(),
    )


@app.get("/predictions/{fixture_id}", response_model=PredictionResponse)
def get_prediction(fixture_id: int) -> PredictionResponse:
    fixture, probs = _predict_prematch(fixture_id)
    return PredictionResponse(
        fixture_id=fixture["fixture_id"],
        league_id=fixture["league_id"],
        starting_at=fixture["starting_at"],
        home_team=fixture["home_name"],
        away_team=fixture["away_name"],
        probabilities=Probabilities(**probs),
        model_version=served.model_version(),
    )


@app.get("/predictions/{fixture_id}/live", response_model=LivePredictionResponse)
def get_live_prediction(fixture_id: int) -> LivePredictionResponse:
    fixture, prior = _predict_prematch(fixture_id)

    live_state = db.get_live_state(fixture_id)
    if live_state is None:
        raise HTTPException(
            status_code=404,
            detail=f"no live state recorded for fixture {fixture_id} - is it in play?",
        )

    return _build_live_response(fixture, prior, live_state)


@app.get("/fixtures/upcoming", response_model=UpcomingFixturesResponse)
def list_upcoming_fixtures() -> UpcomingFixturesResponse:
    """Fixtures kicking off within _UPCOMING_WINDOW, each with the model's
    pre-match prediction where enough history exists, plus market odds and
    the model-vs-market edge where the odds poller has matched a bookmaker
    event to the fixture (PROJECT_SPEC.md Phase 4)."""
    now = datetime.now(timezone.utc)
    rows = db.list_upcoming_fixtures(now, now + _UPCOMING_WINDOW)

    fixtures: list[UpcomingFixture] = []
    for row in rows:
        try:
            features = build_match_features(row["fixture_id"])
        except NotEnoughHistoryError:
            fixtures.append(_fixture_without_prediction(row))
            continue
        except Exception:
            # A single fixture's malformed/incomplete data (e.g. a stale
            # unbackfilled history row) shouldn't 500 the whole list - log
            # and fall back to odds-only, same as NotEnoughHistoryError.
            logger.exception(
                "build_match_features failed for fixture_id=%s", row["fixture_id"]
            )
            fixtures.append(_fixture_without_prediction(row))
            continue

        if features is None:
            continue  # fixture vanished between list and lookup; skip it

        model_probs = served.predict_proba(features)
        market, edge, flagged = _market_and_edge(features.fixture["fixture_id"], model_probs)

        # BTTS rides on v3's own feature vector (app/model_btts.py's
        # docstring), so it needs no NotEnoughHistoryError handling of its
        # own - a fixture that reached here already has one. It is still
        # skipped when the vector is None (SERVED_MODEL_VERSION != "v3", so
        # ratings_state was never replayed) or the BTTS model never loaded.
        btts_probs = btts_market = btts_edge = None
        btts_flagged = False
        if model_btts.is_loaded() and features.v3_vector is not None:
            btts_probs = model_btts.predict_proba(features.v3_vector)
            btts_market, btts_edge, btts_flagged = _btts_market_and_edge(
                features.fixture["fixture_id"], btts_probs
            )

        # probabilities_v2/model_version_v2 predate v2 becoming the serving
        # path and just mirror `probabilities`. They follow the *served* model
        # rather than model_v2 specifically: the frontend reads these fields as
        # "the prediction", and under SERVED_MODEL_VERSION=v3 model_v2 is not
        # even loaded, so naming it here would raise. Drop both once the
        # frontend reads `probabilities`.
        probs_v2 = Probabilities(**model_probs)
        version_v2 = served.model_version()

        fixtures.append(
            UpcomingFixture(
                fixture_id=features.fixture["fixture_id"],
                league_id=features.fixture["league_id"],
                starting_at=features.fixture["starting_at"],
                home_team=features.fixture["home_name"],
                away_team=features.fixture["away_name"],
                probabilities=Probabilities(**model_probs),
                model_version=served.model_version(),
                market=market,
                edge=edge,
                flagged=flagged,
                probabilities_v2=probs_v2,
                model_version_v2=version_v2,
                btts_probabilities=(
                    BTTSProbabilities(**btts_probs) if btts_probs is not None else None
                ),
                btts_market=btts_market,
                btts_edge=btts_edge,
                btts_flagged=btts_flagged,
            )
        )

    fixtures.sort(key=lambda fx: fx.starting_at)

    return UpcomingFixturesResponse(
        generated_at=now, window_days=_UPCOMING_WINDOW.days, fixtures=fixtures
    )


@app.get("/models/comparison", response_model=ModelComparisonResponse)
def get_model_comparison() -> ModelComparisonResponse:
    """The latest offline incumbent-vs-challenger comparison logged by
    ml/player_stats/train_challenger.py. This is validation performance, not
    live predictions, since the challenger isn't served yet."""
    try:
        comparison = model_comparison.latest_availability_comparison()
    except model_comparison.NoComparisonRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return ModelComparisonResponse(
        run_id=comparison["run_id"],
        n_train_rows=comparison["n_train_rows"],
        n_val_rows=comparison["n_val_rows"],
        incumbent=ModelPerformance(
            label="Team form",
            log_loss=comparison["incumbent_log_loss"],
            brier=comparison["incumbent_brier"],
        ),
        challenger=ModelPerformance(
            label="Team form + availability",
            log_loss=comparison["challenger_log_loss"],
            brier=comparison["challenger_brier"],
        ),
        log_loss_improvement=comparison["log_loss_improvement"],
        ci_low=comparison["ci_low"],
        ci_high=comparison["ci_high"],
        p_better=comparison["p_better"],
        clears_promotion_bar=comparison["clears_promotion_bar"],
    )


def _fixture_without_prediction(row) -> UpcomingFixture:
    """Builds an UpcomingFixture with no model prediction (insufficient
    history, or the feature build failed) but still carries market odds
    if available."""
    odds_row = db.get_odds(row["fixture_id"])
    market = (
        Probabilities(
            home_win=odds_row["implied_home"],
            draw=odds_row["implied_draw"],
            away_win=odds_row["implied_away"],
        )
        if odds_row is not None
        else None
    )
    return UpcomingFixture(
        fixture_id=row["fixture_id"],
        league_id=row["league_id"],
        starting_at=row["starting_at"],
        home_team=row["home_name"],
        away_team=row["away_name"],
        probabilities=None,
        model_version=None,
        market=market,
    )


def _market_and_edge(
    fixture_id: int, model_probs: dict
) -> tuple[Probabilities | None, Probabilities | None, bool]:
    """Looks up the fixture's odds match (if any), computes edge = model -
    market per outcome, and persists a snapshot to `edges` for Phase 6's
    backtest module. Returns (None, None, False) when there's no odds match."""
    odds_row = db.get_odds(fixture_id)
    if odds_row is None:
        return None, None, False

    market_probs = {
        "home_win": odds_row["implied_home"],
        "draw": odds_row["implied_draw"],
        "away_win": odds_row["implied_away"],
    }
    edge_probs = {
        outcome: model_probs[outcome] - market_probs[outcome] for outcome in model_probs
    }
    flagged = max(abs(v) for v in edge_probs.values()) >= EDGE_FLAG_THRESHOLD
    prices = {
        "home_win": odds_row["home_price"],
        "draw": odds_row["draw_price"],
        "away_win": odds_row["away_price"],
    }

    db.insert_edge(
        fixture_id, model_probs, market_probs, edge_probs, flagged, prices,
        SERVED_MODEL_VERSION,
    )

    return Probabilities(**market_probs), Probabilities(**edge_probs), flagged


def _btts_market_and_edge(
    fixture_id: int, model_probs: dict
) -> tuple[BTTSProbabilities | None, BTTSProbabilities | None, bool]:
    """_market_and_edge's BTTS sibling: two outcomes instead of three, and
    the odds row's btts_* columns can be None (no quote on the last poll)
    even when the h2h ones are present, so that is checked independently of
    whether db.get_odds() returned a row at all."""
    odds_row = db.get_odds(fixture_id)
    if odds_row is None or odds_row["implied_btts_yes"] is None:
        return None, None, False

    market_probs = {
        "yes": odds_row["implied_btts_yes"],
        "no": odds_row["implied_btts_no"],
    }
    edge_probs = {
        outcome: model_probs[outcome] - market_probs[outcome] for outcome in model_probs
    }
    flagged = max(abs(v) for v in edge_probs.values()) >= EDGE_FLAG_THRESHOLD
    prices = {
        "yes": odds_row["btts_yes_price"],
        "no": odds_row["btts_no_price"],
    }

    db.insert_btts_edge(
        fixture_id,
        model_probs["yes"], model_probs["no"],
        market_probs["yes"], market_probs["no"],
        edge_probs["yes"], edge_probs["no"],
        flagged,
        prices["yes"], prices["no"],
        SERVED_BTTS_MODEL_VERSION,
    )

    return BTTSProbabilities(**market_probs), BTTSProbabilities(**edge_probs), flagged


@app.websocket("/ws/predictions/{fixture_id}/live")
async def stream_live_prediction(websocket: WebSocket, fixture_id: int) -> None:
    """Pushes a new LivePredictionResponse whenever live match state changes
    for `fixture_id`, polling DynamoDB every _LIVE_STREAM_POLL_SECONDS. There's
    no DB change-notification wired up (see PROJECT_SPEC.md Phase 3) so this
    is a poll-and-diff loop rather than a true push."""
    await websocket.accept()

    try:
        fixture, prior = _predict_prematch(fixture_id)
    except HTTPException as exc:
        await websocket.close(code=4004, reason=str(exc.detail))
        return

    last_sent: dict | None = None
    try:
        while True:
            live_state = db.get_live_state(fixture_id)
            if live_state is not None:
                payload = _build_live_response(fixture, prior, live_state).model_dump(
                    mode="json"
                )
                if payload != last_sent:
                    await websocket.send_json(payload)
                    last_sent = payload
            await asyncio.sleep(_LIVE_STREAM_POLL_SECONDS)
    except WebSocketDisconnect:
        pass


def _build_live_response(
    fixture: db.FixtureRow, prior: dict[str, float], live_state: db.LiveStateRow
) -> LivePredictionResponse:
    probs = adjust_for_live_state(
        prior,
        state=live_state["state"],
        home_goals=live_state["home_goals"],
        away_goals=live_state["away_goals"],
    )
    return LivePredictionResponse(
        fixture_id=fixture["fixture_id"],
        league_id=fixture["league_id"],
        starting_at=fixture["starting_at"],
        home_team=fixture["home_name"],
        away_team=fixture["away_name"],
        probabilities=Probabilities(**probs),
        model_version=served.model_version(),
        state=live_state["state"],
        home_goals=live_state["home_goals"],
        away_goals=live_state["away_goals"],
        live_state_updated_at=live_state["updated_at"],
    )


def _predict_prematch(fixture_id: int) -> tuple[db.FixtureRow, dict[str, float]]:
    try:
        features = build_match_features(fixture_id)
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if features is None:
        raise HTTPException(status_code=404, detail=f"fixture {fixture_id} not found")

    return features.fixture, served.predict_proba(features)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )
