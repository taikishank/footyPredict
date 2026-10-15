"""The train/serve equivalence test for app/dc_state.py's Dixon-Coles state,
tests/test_ratings_state.py's own point applied to a second rating: the
(lam, mu, rho) the live replay hands app/model_btts.py's blend must be the
same (lam, mu, rho) ml/build_goals_features.py's offline replay already
validated for the 1X2 task, or the Poisson half of the blend is trained on
one fit and served from another.

Reads the offline archive (results_index.parquet) as ground truth and feeds
the same rows through the service's own loader - no Postgres, no model.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app import dc_state
from build_goals_features import INDEX_PATH, replay_arm


@pytest.fixture(autouse=True)
def _reset_module_state():
    """dc_state's loaded state is module-level globals, same as
    ratings_state.py's - load_state() here must not leak into whichever
    test module pytest happens to run next (tests/test_main.py's own
    fixtures stub app.dc_state.btts_probability for exactly this reason,
    but there is no need to rely on that from this side too)."""
    yield
    dc_state._params = {}
    dc_state._loaded_at = 0.0
    dc_state._n_results = 0
    dc_state._latest_result = None


@pytest.fixture(scope="module")
def archive():
    if not INDEX_PATH.exists():
        pytest.skip("offline goals artifact not built - run ml/build_goals_features.py")
    return pd.read_parquet(INDEX_PATH).sort_values(["date", "fixture_id"]).reset_index(drop=True)


def _as_result_rows(index: pd.DataFrame) -> list[dict]:
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


@pytest.mark.parametrize("cut", [5000, 20000, 39000])
def test_served_rates_match_the_offline_replay(archive, monkeypatch, cut):
    """Fitting everything before fixture N, with `as_of` set to fixture N's
    own date, must reproduce the same (lam, mu, rho)
    build_goals_features.replay_arm() gives fixture N - the arm this serves
    is DC_HALF_LIFE/DC_CADENCE, i.e. build_goals_features.py's own
    DIAGNOSTIC_ARM.

    `as_of` matters here: replay_arm() checks whether *fixture N's own
    date* has crossed the next refit threshold before pricing it, so a
    service that only replays completed fixtures and stops - with no
    `as_of` - can serve a stale fit when N falls far enough past the
    archive's last row. final_params()'s docstring covers why.
    """
    target = archive.iloc[cut]
    rows = _as_result_rows(archive.iloc[:cut])
    all_params = dc_state._fit_all_leagues(rows, as_of=target.date)

    values, league_level = replay_arm(archive, dc_state.DC_HALF_LIFE, dc_state.DC_CADENCE)
    expected_lam, expected_mu = values[cut, 0], values[cut, 1]
    expected_rho = league_level[cut, 0]

    params = all_params[int(target.league_id)]
    lam, mu = params.rates(int(target.home_id), int(target.away_id))

    assert lam == pytest.approx(expected_lam, rel=1e-6)
    assert mu == pytest.approx(expected_mu, rel=1e-6)
    assert params.rho == pytest.approx(expected_rho, rel=1e-6)


def test_btts_probability_is_none_for_an_unfitted_league(archive, monkeypatch):
    cut = 5000
    monkeypatch.setattr(dc_state, "get_results_index", lambda: _as_result_rows(archive.iloc[:cut]))
    dc_state.load_state()

    assert dc_state.btts_probability(league_id=999999, home_id=1, away_id=2) is None


def test_state_summary_reports_what_was_replayed(archive, monkeypatch):
    cut = 1000
    rows = _as_result_rows(archive.iloc[:cut])
    monkeypatch.setattr(dc_state, "get_results_index", lambda: rows)
    dc_state.load_state()

    summary = dc_state.state_summary()
    assert dc_state.is_loaded()
    assert summary["n_results"] == cut
    assert summary["latest_result"] == rows[-1]["starting_at"].isoformat()


def test_refresh_loop_runs_once_when_disabled(archive, monkeypatch):
    """DC_REFRESH_SECONDS <= 0 must still perform the initial load - it
    disables the repeat, not the first replay."""
    monkeypatch.setattr(dc_state, "DC_REFRESH_SECONDS", 0)
    monkeypatch.setattr(dc_state, "get_results_index", lambda: _as_result_rows(archive.iloc[:1000]))
    dc_state._refresh_loop()
    assert dc_state.is_loaded()
