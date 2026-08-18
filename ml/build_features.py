import json
from pathlib import Path

import pandas as pd

from feature_lib import MIN_WINDOW, extract_team_stats, team_form

RAW_DIR = Path(__file__).parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
OUT_PATH = FEATURES_DIR / "2000_window5_8.csv"

FINAL_SCORE_TYPE_ID = 1525
GOALS_STAT_TYPE_ID = 52


def parse_fixture(fixture: dict) -> dict:
    home = next(p for p in fixture["participants"] if p["meta"]["location"] == "home")
    away = next(p for p in fixture["participants"] if p["meta"]["location"] == "away")

    final_scores = {
        s["score"]["participant"]: s["score"]["goals"]
        for s in fixture["scores"]
        if s["type_id"] == FINAL_SCORE_TYPE_ID
    }
    if "home" in final_scores and "away" in final_scores:
        home_goals, away_goals = final_scores["home"], final_scores["away"]
    else:
        # Fall back to the goals stat instead of failing outright - some
        # fixtures apparently don't carry a CURRENT score entry.
        goals_by_participant = {
            s["participant_id"]: s["data"]["value"]
            for s in fixture["statistics"]
            if s["type_id"] == GOALS_STAT_TYPE_ID
        }
        if home["id"] not in goals_by_participant or away["id"] not in goals_by_participant:
            # Fixture hasn't been played yet (or was postponed/cancelled) -
            # no result to train on, so skip it.
            return None
        home_goals = goals_by_participant[home["id"]]
        away_goals = goals_by_participant[away["id"]]

    if home_goals > away_goals:
        result = "home_win"
    elif away_goals > home_goals:
        result = "away_win"
    else:
        result = "draw"

    return {
        "fixture_id": fixture["id"],
        "league_id": fixture["league_id"],
        "date": fixture["starting_at"],
        "home_id": home["id"],
        "away_id": away["id"],
        "home_name": home["name"],
        "away_name": away["name"],
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": result,
        "home_stats": extract_team_stats(fixture, home["id"]),
        "away_stats": extract_team_stats(fixture, away["id"]),
    }


def load_fixtures() -> list[dict]:
    # raw/all_fixtures/*.json now covers the full subscription - the old
    # per-league raw dirs are a subset of it, so reading only this one
    # avoids double-counting fixtures that exist in both.
    fixtures = [
        parsed
        for p in (RAW_DIR / "all_fixtures").glob("*.json")
        if (parsed := parse_fixture(json.loads(p.read_text()))) is not None
    ]
    fixtures.sort(key=lambda f: f["date"])
    return fixtures


def build_feature_table() -> pd.DataFrame:
    fixtures = load_fixtures()
    team_history: dict[int, list[dict]] = {}
    rows = []

    for fx in fixtures:
        home_hist = team_history.get(fx["home_id"], [])
        away_hist = team_history.get(fx["away_id"], [])

        if len(home_hist) >= MIN_WINDOW and len(away_hist) >= MIN_WINDOW:
            row = {
                "fixture_id": fx["fixture_id"],
                "date": fx["date"],
                "home_name": fx["home_name"],
                "away_name": fx["away_name"],
                "result": fx["result"],
            }
            for name, value in team_form(home_hist).items():
                row[f"home_form_{name}"] = value
            for name, value in team_form(away_hist).items():
                row[f"away_form_{name}"] = value
            rows.append(row)

        team_history.setdefault(fx["home_id"], []).append(
            {"goals_for": fx["home_goals"], "goals_against": fx["away_goals"], "stats": fx["home_stats"]}
        )
        team_history.setdefault(fx["away_id"], []).append(
            {"goals_for": fx["away_goals"], "goals_against": fx["home_goals"], "stats": fx["away_stats"]}
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = build_feature_table()
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
