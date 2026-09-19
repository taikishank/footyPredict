"""The train/serve equivalence test for v3's rating columns.

The point of app/ratings_state.py is that the ratings the service predicts
from are the ratings the model was fitted on. That is a claim about two
different code paths - ml/build_rating_features.py's replay over a parquet and
the service's replay over Postgres - agreeing to the last decimal, and it is
worth asserting rather than reasoning about: ml/FINDINGS_SCOPE.md records what
it cost the last time a serving path quietly diverged from a training one.

These tests read the offline artifacts (results_index.parquet, ratings.csv) as
the ground truth and feed the same rows through the service's own loader, so
they need no Postgres and no model.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app import ratings_state
from build_rating_features import INDEX_PATH, order
from ratings_lib import ARMS

RATINGS_CSV = INDEX_PATH.parent / "ratings.csv"


@pytest.fixture(scope="module")
def archive():
    if not INDEX_PATH.exists() or not RATINGS_CSV.exists():
        pytest.skip("offline rating artifacts not built - run ml/build_rating_features.py")
    index = order(pd.read_parquet(INDEX_PATH))
    ratings = pd.read_csv(RATINGS_CSV).set_index("fixture_id")
    return index, ratings


def _load_through_service(rows, monkeypatch):
    """Drive ratings_state.load_state() off an in-memory result list."""
    monkeypatch.setattr(ratings_state, "get_results_index", lambda: rows)
    ratings_state.load_state()


def _as_result_rows(index):
    return [
        {
            "league_id": int(fx.league_id),
            "starting_at": fx.date.to_pydatetime(),
            "home_id": int(fx.home_id),
            "away_id": int(fx.away_id),
            "home_goals": int(fx.home_goals),
            "away_goals": int(fx.away_goals),
        }
        for fx in index.itertuples(index=False)
    ]


@pytest.mark.parametrize("cut", [20000, 30000, 39000])
def test_served_ratings_match_the_training_row(archive, monkeypatch, cut):
    """Replaying everything before fixture N must reproduce ratings.csv's row
    for fixture N - which is what the model was trained on."""
    index, ratings = archive
    target = index.iloc[cut]
    _load_through_service(_as_result_rows(index.iloc[:cut]), monkeypatch)

    served = ratings_state.rating_features(
        int(target.league_id), int(target.home_id), int(target.away_id),
        target.date.to_pydatetime(),
    )
    expected = ratings.loc[int(target.fixture_id)]

    for name in ARMS[ratings_state.ARM].FIELD_ORDER:
        assert served[name] == pytest.approx(float(expected[name]), abs=1e-9), name


def test_field_order_matches_the_training_column_order(archive):
    """v3's vector is built by indexing FIELD_ORDER, so a reordering here would
    silently feed the model transposed columns."""
    _, ratings = archive
    assert ratings_state.FIELD_ORDER == ARMS["atkdef"].FIELD_ORDER
    for name in ratings_state.FIELD_ORDER:
        assert name in ratings.columns


def test_serving_does_not_mutate_the_replayed_state(archive, monkeypatch):
    """rating_features() rolls a copy forward. If it rolled the singleton, the
    first request of a new season would regress every rating in the league and
    every later request would read the damaged state."""
    index, _ = archive
    cut = 30000
    _load_through_service(_as_result_rows(index.iloc[:cut]), monkeypatch)
    target = index.iloc[cut]

    # A fixture four seasons past the archive's end forces season_boundary()
    # and recenter() to fire on whatever state they are handed.
    far_future = target.date.to_pydatetime() + timedelta(days=4 * 365)
    args = (int(target.league_id), int(target.home_id), int(target.away_id))

    before = ratings_state.rating_features(*args, target.date.to_pydatetime())
    ratings_state.rating_features(*args, far_future)
    after = ratings_state.rating_features(*args, target.date.to_pydatetime())

    assert before == after


def test_rolling_forward_is_idempotent(archive, monkeypatch):
    """Two requests for the same upcoming fixture must price it identically -
    the second must not regress the ratings a second time."""
    index, _ = archive
    cut = 30000
    _load_through_service(_as_result_rows(index.iloc[:cut]), monkeypatch)
    target = index.iloc[cut]
    future = target.date.to_pydatetime() + timedelta(days=400)
    args = (int(target.league_id), int(target.home_id), int(target.away_id))

    assert ratings_state.rating_features(*args, future) == \
        ratings_state.rating_features(*args, future)


def test_state_summary_reports_what_was_replayed(archive, monkeypatch):
    index, _ = archive
    rows = _as_result_rows(index.iloc[:1000])
    _load_through_service(rows, monkeypatch)

    summary = ratings_state.state_summary()
    assert ratings_state.is_loaded()
    assert summary["arm"] == "atkdef"
    assert summary["n_results"] == 1000
    assert summary["latest_result"] == rows[-1]["starting_at"].isoformat()


def test_unloaded_state_raises_rather_than_serving_zeros(monkeypatch):
    """A rating of 0.0 is a legitimate value (the columns are league-centred),
    so an unloaded state must not be allowed to look like a neutral team."""
    monkeypatch.setattr(ratings_state, "_replay", None)
    with pytest.raises(RuntimeError, match="not loaded"):
        ratings_state.rating_features(8, 1, 2, datetime(2026, 1, 1, tzinfo=timezone.utc))
