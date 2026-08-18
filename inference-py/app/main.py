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
from app.schemas import HealthResponse, PredictionResponse, Probabilities


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
    try:
        features = build_match_features(fixture_id)
    except NotEnoughHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if features is None:
        raise HTTPException(status_code=404, detail=f"fixture {fixture_id} not found")

    probs = model.predict_proba(features.vector)
    fixture = features.fixture
    return PredictionResponse(
        fixture_id=fixture["fixture_id"],
        league_id=fixture["league_id"],
        starting_at=fixture["starting_at"],
        home_team=fixture["home_name"],
        away_team=fixture["away_name"],
        probabilities=Probabilities(**probs),
        model_version=model.model_version(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(Path(__file__).resolve().parent.parent),
    )
