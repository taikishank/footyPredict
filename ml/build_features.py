import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "data" / "raw"
FEATURES_DIR = Path(__file__).parent / "data" / "features"
OUT_PATH = FEATURES_DIR / "all_leagues_features_removed_tackles.csv"

MIN_WINDOW = 3
MAX_WINDOW = 5

FINAL_SCORE_TYPE_ID = 1525
GOALS_STAT_TYPE_ID = 52

# type_id -> feature name
STAT_IDS = {
    42: "shots_total",
    86: "shots_on_target",
    41: "shots_off_target",
    58: "shots_blocked",
    64: "hit_woodwork",
    47: "penalties",
    80: "passes",
    82: "successful_passes_pct",
    62: "long_passes",
    27264: "successful_long_passes",
    27265: "successful_long_passes_pct",
    34: "corners",
    78: "tackles",
    # 100: "interceptions",
    45: "possession",
    51: "offsides",
    99: "accurate_crosses",
    98: "total_crosses",
    56: "fouls",
    84: "yellow_cards",
    83: "red_cards",
    57: "saves",
    580: "big_chances_created",
    581: "big_chances_missed",
}


def extract_team_stats(fixture: dict, participant_id: int) -> dict:
    stats = {name: 0 for name in STAT_IDS.values()}
    for entry in fixture["statistics"]:
        if entry["participant_id"] != participant_id:
            continue
        name = STAT_IDS.get(entry["type_id"])
        if name is not None:
            stats[name] = entry["data"]["value"]

    stats["shot_pct"] = (
        stats["shots_on_target"] / stats["shots_total"] if stats["shots_total"] else 0.0
    )
    chances = stats["big_chances_created"] + stats["big_chances_missed"]
    stats["big_chances_conversion_rate"] = (
        stats["big_chances_created"] / chances if chances else 0.0
    )
    return stats


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
    # RAW_DIR holds one subdirectory per league (raw/premier_league/*.json,
    # raw/la_liga/*.json, ...) - glob across all of them.
    fixtures = [parse_fixture(json.loads(p.read_text())) for p in RAW_DIR.glob("*/*.json")]
    fixtures.sort(key=lambda f: f["date"])
    return fixtures


def team_form(history: list[dict]) -> dict:
    window = history[-MAX_WINDOW:]
    n = len(window)
    form = {"goals_for": sum(h["goals_for"] for h in window) / n,
            "goals_against": sum(h["goals_against"] for h in window) / n}
    for name in list(STAT_IDS.values()) + ["shot_pct", "big_chances_conversion_rate"]:
        form[name] = sum(h["stats"][name] for h in window) / n
    return form


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
