if __name__ == "__main__":
    # Allow `python3 app/main.py` / `python3 inference-py/app/main.py` to work
    # regardless of cwd - running this file directly only puts its own
    # directory (app/) on sys.path, not inference-py/, so `import app` fails.
    # `uvicorn app.main:app` from inference-py/ doesn't need this.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app import db, model
from app.config import CORS_ALLOWED_ORIGINS, EDGE_FLAG_THRESHOLD
from app.features import NotEnoughHistoryError, build_match_features
from app.live_prob import adjust_for_live_state
from app.schemas import (
    HealthResponse,
    LivePredictionResponse,
    PredictionResponse,
    Probabilities,
    UpcomingFixture,
    UpcomingFixturesResponse,
)

# How often the WS stream re-reads live_match_state and re-sends if changed.
# ingestor-go's live poller writes to that table on its own interval (see
# ingestor-go/internal/live/service.go); this just needs to be frequent
# enough that dashboard updates feel live without hammering Postgres.
_LIVE_STREAM_POLL_SECONDS = 4

# Matches the dashboard's Upcoming tab window (PROJECT_SPEC.md Phase 4).
_UPCOMING_WINDOW = timedelta(days=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open()
    model.load_model()
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
        status="ok" if model.is_loaded() else "model not loaded",
        model_version=model.model_version() if model.is_loaded() else None,
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
        model_version=model.model_version(),
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
            fixtures.append(
                UpcomingFixture(
                    fixture_id=row["fixture_id"],
                    league_id=row["league_id"],
                    starting_at=row["starting_at"],
                    home_team=row["home_name"],
                    away_team=row["away_name"],
                    probabilities=None,
                    model_version=None,
                    market=market,
                )
            )
            continue

        if features is None:
            continue  # fixture vanished between list and lookup; skip it

        model_probs = model.predict_proba(features.vector)
        market, edge, flagged = _market_and_edge(features.fixture["fixture_id"], model_probs)

        fixtures.append(
            UpcomingFixture(
                fixture_id=features.fixture["fixture_id"],
                league_id=features.fixture["league_id"],
                starting_at=features.fixture["starting_at"],
                home_team=features.fixture["home_name"],
                away_team=features.fixture["away_name"],
                probabilities=Probabilities(**model_probs),
                model_version=model.model_version(),
                market=market,
                edge=edge,
                flagged=flagged,
            )
        )

    return UpcomingFixturesResponse(
        generated_at=now, window_days=_UPCOMING_WINDOW.days, fixtures=fixtures
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

    db.insert_edge(fixture_id, model_probs, market_probs, edge_probs, flagged)

    return Probabilities(**market_probs), Probabilities(**edge_probs), flagged


@app.websocket("/ws/predictions/{fixture_id}/live")
async def stream_live_prediction(websocket: WebSocket, fixture_id: int) -> None:
    """Pushes a new LivePredictionResponse whenever live_match_state changes
    for `fixture_id`, polling Postgres every _LIVE_STREAM_POLL_SECONDS. There's
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
        model_version=model.model_version(),
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

    return features.fixture, model.predict_proba(features.vector)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )
