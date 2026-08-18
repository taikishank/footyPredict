from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, main, model
from app.features import MatchFeatures, NotEnoughHistoryError

FIXTURE = {
    "fixture_id": 123,
    "league_id": 8,
    "starting_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    "home_id": 1,
    "away_id": 2,
    "home_name": "Arsenal",
    "away_name": "Chelsea",
}


@pytest.fixture(autouse=True)
def _no_real_startup(monkeypatch):
    # lifespan opens a real db pool / loads the model from disk-or-S3; stub
    # both out so the client fixture doesn't need live infra.
    monkeypatch.setattr(db.pool, "open", lambda: None)
    monkeypatch.setattr(db.pool, "close", lambda: None)
    monkeypatch.setattr(model, "load_model", lambda: None)
    monkeypatch.setattr(model, "is_loaded", lambda: True)
    monkeypatch.setattr(model, "model_version", lambda: "test-model@1")


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_version": "test-model@1"}


def test_get_prediction_returns_probabilities(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: MatchFeatures(fixture=FIXTURE, vector=np.zeros((1, 54))),
    )
    monkeypatch.setattr(
        model, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    )

    resp = client.get("/predictions/123")

    assert resp.status_code == 200
    body = resp.json()
    assert body["fixture_id"] == 123
    assert body["home_team"] == "Arsenal"
    assert body["probabilities"] == {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}


def test_get_prediction_404_when_fixture_missing(client, monkeypatch):
    monkeypatch.setattr(main, "build_match_features", lambda fixture_id: None)
    resp = client.get("/predictions/999")
    assert resp.status_code == 404


def test_get_prediction_422_when_not_enough_history(client, monkeypatch):
    def raise_not_enough(fixture_id):
        raise NotEnoughHistoryError(team_id=1, n_available=2)

    monkeypatch.setattr(main, "build_match_features", raise_not_enough)
    resp = client.get("/predictions/123")
    assert resp.status_code == 422
