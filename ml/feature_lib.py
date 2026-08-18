"""Team-form feature engineering, shared between offline training
(build_features.py) and the FastAPI inference service
(inference-py/app/features.py) so the two stay in lockstep.
"""

MIN_WINDOW = 5
MAX_WINDOW = 8

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

# Exact key order team_form() below produces - callers that need to build a
# feature vector by hand (rather than off a DataFrame) rely on this order
# matching the columns the model was trained on (home_form_<x>/away_form_<x>).
FORM_FIELD_ORDER = (
    ["goals_for", "goals_against"]
    + list(STAT_IDS.values())
    + ["shot_pct", "big_chances_conversion_rate"]
)


def extract_team_stats(fixture: dict, participant_id: int) -> dict:
    stats = {name: 0 for name in STAT_IDS.values()}
    for entry in fixture["statistics"]:
        if entry["participant_id"] != participant_id:
            continue
        name = STAT_IDS.get(entry["type_id"])
        if name is not None and entry["data"]["value"] is not None:
            stats[name] = entry["data"]["value"]

    stats["shot_pct"] = (
        stats["shots_on_target"] / stats["shots_total"] if stats["shots_total"] else 0.0
    )
    chances = stats["big_chances_created"] + stats["big_chances_missed"]
    stats["big_chances_conversion_rate"] = (
        stats["big_chances_created"] / chances if chances else 0.0
    )
    return stats


def team_form(history: list[dict]) -> dict:
    """history: most-recent-last list of {"goals_for", "goals_against", "stats"}."""
    window = history[-MAX_WINDOW:]
    n = len(window)
    form = {
        "goals_for": sum(h["goals_for"] for h in window) / n,
        "goals_against": sum(h["goals_against"] for h in window) / n,
    }
    for name in list(STAT_IDS.values()) + ["shot_pct", "big_chances_conversion_rate"]:
        form[name] = sum(h["stats"][name] for h in window) / n
    return form
