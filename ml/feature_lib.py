"""Team-form feature engineering, shared between offline training
(build_features.py) and the FastAPI inference service
(inference-py/app/features.py) so the two stay in lockstep.
"""

MIN_WINDOW = 3
MAX_WINDOW = 8

# A club's goals scored over its last GOALS_WINDOW matches, normalised by its
# league's recent level (club_goals_relative below) - the single largest
# squad-stats finding outside the availability block in
# ml/player_stats/FINDINGS_PHASE9.md (+0.0035 log loss), and the one that
# needs no lineup data at all, only goals per club per match. Ported here
# (rather than depended on from ml/player_stats/player_feature_lib.py) since
# this is the shared team-form/inference module and the player-stats work is
# a separate, still-offline experiment track.
GOALS_WINDOW = 38


class LeagueRollingMean:
    """A league's recent mean of some per-club quantity, as of now.

    Goals scored over GOALS_WINDOW matches inherits the archive's density
    ramp (more leagues/seasons covered as the years go on) the same way raw
    career-appearance counts did in ml/player_stats/FINDINGS_PHASE9.md -
    dividing by a rolling reference turns a count into "how does this club's
    scoring compare with the league it's actually in right now", which has no
    time trend. Rolling rather than expanding so it tracks the ramp instead
    of lagging behind it.
    """

    WINDOW = 200

    def __init__(self) -> None:
        self._recent: dict[int, list[float]] = {}

    def update(self, league_id: int, value: float) -> None:
        recent = self._recent.setdefault(league_id, [])
        recent.append(value)
        if len(recent) > self.WINDOW:
            del recent[: -self.WINDOW]

    def reference(self, league_id: int) -> float | None:
        recent = self._recent.get(league_id)
        return sum(recent) / len(recent) if recent else None


def club_goals_relative(volume: float, league_reference: float | None) -> float | None:
    """A club's GOALS_WINDOW-match goal volume against its league's recent
    level, or None before a reference exists (a league's very first
    fixtures)."""
    return volume / league_reference if league_reference else None

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


def team_form(history: list[dict], window_size: int = MAX_WINDOW) -> dict:
    """history: most-recent-last list of {"goals_for", "goals_against", "stats"}.

    window_size defaults to MAX_WINDOW - the served recipe. It is a parameter
    only so the offline sweep (ml/build_window_features.py) can replay the
    archive at several window lengths in one pass; training and inference both
    take the default, so the two stay in lockstep as before.
    """
    window = history[-window_size:]
    n = len(window)
    form = {
        "goals_for": sum(h["goals_for"] for h in window) / n,
        "goals_against": sum(h["goals_against"] for h in window) / n,
    }
    for name in list(STAT_IDS.values()) + ["shot_pct", "big_chances_conversion_rate"]:
        form[name] = sum(h["stats"][name] for h in window) / n
    return form
