from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import db, main, model_comparison, ratings_state, served
from app.config import SERVED_MODEL_VERSION
from app.features import MatchFeatures, NotEnoughHistoryError

#: What ratings_state reports to /health when the served model needs it. Only
#: read when served.NEEDS_RATINGS; the values are arbitrary but the shape is
#: ratings_state.state_summary()'s.
RATINGS_SUMMARY = {
    "arm": "atkdef", "n_results": 40110,
    "latest_result": "2026-09-13T16:30:00+00:00", "age_seconds": 0.0,
}

#: Column counts per vector, matching app/features.py's MatchFeatures: 54 form
#: columns, +2 club_goals_relative for v2, +7 atkdef ratings for v3.
VECTOR_WIDTHS = {"vector": 54, "v2_vector": 56, "v3_vector": 63}


def match_features(fixture=None) -> MatchFeatures:
    """A MatchFeatures whose *served* vector is populated.

    Tests care that the request handler gets a row the served model can eat,
    not which column count that is. Building it from served.VECTOR_ATTR means
    a promotion does not require editing every test that stubs a prediction -
    which was the bulk of the v2 -> v3 churn.
    """
    vectors = {name: None for name in VECTOR_WIDTHS}
    vectors["vector"] = np.zeros((1, VECTOR_WIDTHS["vector"]))
    vectors[served.VECTOR_ATTR] = np.zeros((1, VECTOR_WIDTHS[served.VECTOR_ATTR]))
    return MatchFeatures(fixture=fixture or FIXTURE, **vectors)


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
    # Stub whichever module SERVED_MODEL_VERSION names, rather than naming one.
    # The artifact may genuinely exist on disk (ml/train_*.py writes one), so
    # loading is stubbed to isolate tests from local dev state.
    #
    # This used to hardcode model_v2 and assert the default was "v2", which
    # caught the v2 -> v3 promotion loudly but then had to be rewritten by
    # hand. Following served.MODULE cannot go stale: the next promotion moves
    # these stubs with it.
    monkeypatch.setattr(served.MODULE, "load_model", lambda: None)
    monkeypatch.setattr(served.MODULE, "is_loaded", lambda: True)
    monkeypatch.setattr(served.MODULE, "model_version", lambda: "test-model@1")
    # A model whose input is more than a Postgres query (v3's replayed ratings)
    # also needs its state stubbed, or served.load() reads the real archive.
    if served.NEEDS_RATINGS:
        monkeypatch.setattr(ratings_state, "load_state", lambda: None)
        monkeypatch.setattr(ratings_state, "is_loaded", lambda: True)
        monkeypatch.setattr(ratings_state, "state_summary", lambda: RATINGS_SUMMARY)


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    # `ratings` is present only for a model whose input is more than a Postgres
    # query - app/served.py replays rating state for v3 and not for v1/v2, and
    # /health reporting "ok" without it would be a lie for v3: the model would
    # be loaded and unable to predict.
    assert resp.json() == {
        "status": "ok",
        "model_version": "test-model@1",
        "ratings": RATINGS_SUMMARY if served.NEEDS_RATINGS else None,
    }


def test_get_prediction_returns_probabilities(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: match_features(),
    )
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
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


def test_get_live_prediction_returns_adjusted_probabilities(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: match_features(),
    )
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    )
    monkeypatch.setattr(
        db,
        "get_live_state",
        lambda fixture_id: {
            "state": "2H",
            "home_goals": 1,
            "away_goals": 0,
            "updated_at": datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc),
        },
    )

    resp = client.get("/predictions/123/live")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "2H"
    assert body["home_goals"] == 1
    assert body["probabilities"]["home_win"] > 0.5


def test_get_live_prediction_404_when_fixture_not_live(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: match_features(),
    )
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    )
    monkeypatch.setattr(db, "get_live_state", lambda fixture_id: None)

    resp = client.get("/predictions/123/live")
    assert resp.status_code == 404


def test_ws_live_prediction_streams_on_state_change(client, monkeypatch):
    monkeypatch.setattr(main, "_LIVE_STREAM_POLL_SECONDS", 0)
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: match_features(),
    )
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    )

    states = [
        {
            "state": "1H",
            "home_goals": 0,
            "away_goals": 0,
            "updated_at": datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc),
        },
        {
            "state": "1H",
            "home_goals": 1,
            "away_goals": 0,
            "updated_at": datetime(2026, 1, 1, 15, 10, tzinfo=timezone.utc),
        },
    ]
    # Repeats the final state forever - the WS loop polls until the client
    # disconnects, so it will ask for more states than we have transitions.
    call_count = {"n": 0}

    def _next_state(fixture_id):
        i = min(call_count["n"], len(states) - 1)
        call_count["n"] += 1
        return states[i]

    monkeypatch.setattr(db, "get_live_state", _next_state)

    with client.websocket_connect("/ws/predictions/123/live") as ws:
        first = ws.receive_json()
        second = ws.receive_json()

    assert first["home_goals"] == 0
    assert second["home_goals"] == 1
    assert second["probabilities"]["home_win"] > first["probabilities"]["home_win"]


def test_list_upcoming_fixtures(client, monkeypatch):
    other_fixture = {**FIXTURE, "fixture_id": 456, "home_name": "Villa", "away_name": "Spurs"}
    monkeypatch.setattr(
        db, "list_upcoming_fixtures", lambda start, end: [FIXTURE, other_fixture]
    )

    def _build_match_features(fixture_id):
        if fixture_id == 456:
            raise NotEnoughHistoryError(team_id=2, n_available=1)
        return match_features()

    monkeypatch.setattr(main, "build_match_features", _build_match_features)
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.5, "draw": 0.3, "away_win": 0.2}
    )
    monkeypatch.setattr(db, "get_odds", lambda fixture_id: None)

    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 200
    body = resp.json()

    assert body["window_days"] == 3
    assert [f["fixture_id"] for f in body["fixtures"]] == [123, 456]
    assert body["fixtures"][0]["probabilities"]["home_win"] == 0.5
    assert body["fixtures"][0]["market"] is None
    assert body["fixtures"][0]["edge"] is None
    assert body["fixtures"][0]["flagged"] is False
    assert body["fixtures"][1]["probabilities"] is None
    assert body["fixtures"][1]["model_version"] is None
    # probabilities_v2/model_version_v2 predate v2 becoming the serving path
    # and now just mirror the served fields (see main.list_upcoming_fixtures);
    # they stay populated until the frontend stops reading them.
    assert body["fixtures"][0]["probabilities_v2"] == body["fixtures"][0]["probabilities"]
    assert body["fixtures"][0]["model_version_v2"] == "test-model@1"


def test_list_upcoming_fixtures_legacy_v2_fields_mirror_served(client, monkeypatch):
    """probabilities_v2/model_version_v2 carry the *served* model, not v2.

    The names predate v2 becoming the serving path and are wrong in both
    halves now that v3 serves - see app/schemas.py. This pins the behaviour
    main.py deliberately implements so the fields cannot quietly start
    reporting a model nobody is grading, and so deleting them once the
    frontend stops reading them is a visible change.
    """
    monkeypatch.setattr(db, "list_upcoming_fixtures", lambda start, end: [FIXTURE])
    monkeypatch.setattr(main, "build_match_features", lambda fixture_id: match_features())
    monkeypatch.setattr(db, "get_odds", lambda fixture_id: None)
    monkeypatch.setattr(
        served.MODULE,
        "predict_proba",
        lambda vector: {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
    )
    monkeypatch.setattr(served.MODULE, "model_version", lambda: f"served-{SERVED_MODEL_VERSION}@1")

    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 200
    fx = resp.json()["fixtures"][0]

    assert fx["probabilities_v2"] == {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}
    assert fx["probabilities_v2"] == fx["probabilities"]
    assert fx["model_version_v2"] == f"served-{SERVED_MODEL_VERSION}@1"
    assert fx["model_version_v2"] == fx["model_version"]


def test_list_upcoming_fixtures_shows_odds_despite_not_enough_history(client, monkeypatch):
    monkeypatch.setattr(db, "list_upcoming_fixtures", lambda start, end: [FIXTURE])

    def _raise_not_enough(fixture_id):
        raise NotEnoughHistoryError(team_id=1, n_available=2)

    monkeypatch.setattr(main, "build_match_features", _raise_not_enough)
    monkeypatch.setattr(
        db,
        "get_odds",
        lambda fixture_id: {
            "implied_home": 0.42, "implied_draw": 0.31, "implied_away": 0.27,
            "home_price": 2.3, "draw_price": 3.4, "away_price": 4.1,
        },
    )

    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 200
    body = resp.json()

    fixture = body["fixtures"][0]
    assert fixture["probabilities"] is None
    assert fixture["model_version"] is None
    assert fixture["market"] == {"home_win": 0.42, "draw": 0.31, "away_win": 0.27}
    assert fixture["edge"] is None
    assert fixture["flagged"] is False


def test_list_upcoming_fixtures_merges_odds_and_flags_edge(client, monkeypatch):
    monkeypatch.setattr(db, "list_upcoming_fixtures", lambda start, end: [FIXTURE])
    monkeypatch.setattr(
        main,
        "build_match_features",
        lambda fixture_id: match_features(),
    )
    monkeypatch.setattr(
        served.MODULE, "predict_proba", lambda vector: {"home_win": 0.6, "draw": 0.25, "away_win": 0.15}
    )
    monkeypatch.setattr(
        db,
        "get_odds",
        lambda fixture_id: {
            "implied_home": 0.5, "implied_draw": 0.28, "implied_away": 0.22,
            "home_price": 2.0, "draw_price": 3.6, "away_price": 4.5,
        },
    )
    inserted = {}
    monkeypatch.setattr(
        db,
        "insert_edge",
        lambda fixture_id, model_probs, market_probs, edge, flagged, prices, model_version: (
            inserted.update(
                fixture_id=fixture_id, flagged=flagged, prices=prices,
                model_probs=model_probs, model_version=model_version,
            )
        ),
    )

    resp = client.get("/fixtures/upcoming")
    assert resp.status_code == 200
    fx = resp.json()["fixtures"][0]

    assert fx["market"] == {"home_win": 0.5, "draw": 0.28, "away_win": 0.22}
    assert fx["edge"]["home_win"] == pytest.approx(0.1)
    # |0.1| >= EDGE_FLAG_THRESHOLD (0.05 default), so this should be flagged.
    assert fx["flagged"] is True
    # The snapshot persisted for grading must be the served (v2) model's
    # probabilities, tagged as such - backtest/grade_calls.py grades strictly
    # one model_version, so a mislabelled row would silently corrupt the
    # v2 record with a v1 prediction.
    assert inserted == {
        "fixture_id": 123,
        "flagged": True,
        "prices": {"home_win": 2.0, "draw": 3.6, "away_win": 4.5},
        "model_probs": {"home_win": 0.6, "draw": 0.25, "away_win": 0.15},
        "model_version": SERVED_MODEL_VERSION,
    }


def test_get_model_comparison(client, monkeypatch):
    monkeypatch.setattr(
        model_comparison,
        "latest_availability_comparison",
        lambda: {
            "run_id": "abc123",
            "n_train_rows": 28362,
            "n_val_rows": 7091,
            "incumbent_log_loss": 0.99979,
            "incumbent_brier": 0.59675,
            "challenger_log_loss": 0.98922,
            "challenger_brier": 0.58944,
            "log_loss_improvement": 0.01057,
            "ci_low": 0.00604,
            "ci_high": 0.01520,
            "p_better": 1.0,
            "clears_promotion_bar": True,
        },
    )

    resp = client.get("/models/comparison")

    assert resp.status_code == 200
    body = resp.json()
    assert body["incumbent"] == {"label": "Team form", "log_loss": 0.99979, "brier": 0.59675}
    assert body["challenger"] == {
        "label": "Team form + availability",
        "log_loss": 0.98922,
        "brier": 0.58944,
    }
    assert body["clears_promotion_bar"] is True


def test_get_model_comparison_404_when_no_run_logged(client, monkeypatch):
    def _raise():
        raise model_comparison.NoComparisonRunError("no availability-tier run logged yet")

    monkeypatch.setattr(model_comparison, "latest_availability_comparison", _raise)

    resp = client.get("/models/comparison")

    assert resp.status_code == 404


def test_ws_live_prediction_closes_when_fixture_missing(client, monkeypatch):
    monkeypatch.setattr(main, "build_match_features", lambda fixture_id: None)

    with client.websocket_connect("/ws/predictions/999/live") as ws:
        with pytest.raises(Exception):
            ws.receive_json()
