if __name__ == "__main__":
    # Allow `python3 app/main.py` / `python3 inference-py/app/main.py` to work
    # regardless of cwd - running this file directly only puts its own
    # directory (app/) on sys.path, not inference-py/, so `import app` fails.
    # `uvicorn app.main:app` from inference-py/ doesn't need this.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import db, model
from app.features import NotEnoughHistoryError, build_match_features
from app.live_prob import adjust_for_live_state
from app.schemas import (
    HealthResponse,
    LivePredictionResponse,
    PredictionResponse,
    Probabilities,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.pool.open()
    model.load_model()
    yield
    db.pool.close()


app = FastAPI(title="LiveEdge Inference", lifespan=lifespan)


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
