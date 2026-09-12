"""Player-level feature engineering for the Phase 9 challenger model.

Deliberately separate from `ml/feature_lib.py`: that module is imported by the
live inference service (`inference-py/app/features.py`) and defines the
incumbent's feature contract, so the incumbent's training input has to stay
byte-identical while the challenger is being evaluated. Nothing here touches
it. If the challenger is promoted (Phase 11-12), the merge happens then.

Two tiers, because per-player stats arrive at different times in the archive
(FINDINGS_DATA_AVAILABILITY.md, "Historical depth"):

  availability (2006+)  minutes-played only: who is in the XI, how settled it
                        is, who is missing. Works across ~35k fixtures.
  performance  (2016+)  ratings and per-90 output by position group: how good
                        the players in it actually are. Works across ~18k.

Keeping them separable is what lets Phase 9 ask two honest questions instead of
one confounded one - does knowing *who plays* help, and does knowing *how well
they play* add anything on top - rather than testing a single block against a
baseline trained on twice the data.

Everything is a pure function over already-accumulated history, in the same
style as `team_form()`: the caller walks fixtures in chronological order and is
responsible for only ever passing in matches that precede the one being
predicted.
"""

import math

# Appearances, not fixtures. A rotated or injured player accumulates history
# far more slowly than his team does (spec section 5), so this window is both
# longer than MAX_WINDOW = 8 and measured on a different clock.
PLAYER_WINDOW = 10

# Below this many prior appearances a player's own form is too noisy to use and
# he is imputed from the league-and-position average instead. Debutants and new
# signings hit this constantly, and a promoted club fields eleven of them at
# once every August, so this is a main path rather than an edge case.
MIN_APPEARANCES = 3

# Team sheets looked at when deciding what this club's "settled" XI is.
MODAL_XI_WINDOW = 5

# How many of a squad's most-used players count as the ones whose absence is
# worth a feature.
KEY_PLAYER_COUNT = 11

POSITION_GROUPS = ("goalkeeper", "defender", "midfielder", "attacker")

# Stats averaged per 90 minutes over a player's window. Rating is handled
# separately - it is already a per-match figure and averaging it per-90 would
# be meaningless.
PER90_STATS = (
    "goals",
    "assists",
    "shots_total",
    "shots_on_target",
    "key_passes",
    "chances_created",
    "big_chances_created",
    "tackles",
    "interceptions",
    "clearances",
    "duels_won",
    "successful_dribbles",
    "saves",
    "goals_conceded",
)

PLAYER_FORM_FIELDS = ("rating", "minutes_per_appearance") + PER90_STATS


def numeric(value) -> float | None:
    """Coerce a cached stat to a number, or None when it was never recorded.

    Worth a named helper rather than an inline `or 0`: the parquet carries
    missing ratings and minutes as NaN, and `NaN or 0` evaluates to NaN in
    Python because NaN is truthy. That silently propagates through every sum
    downstream and turns whole feature blocks into NaN, which is exactly the
    kind of bug that reads as "the data is thin here" instead of "the code is
    wrong".
    """
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return None if math.isnan(value) else value


def player_form(history: list[dict]) -> dict:
    """Rolling form over a player's last PLAYER_WINDOW *appearances*.

    `history` is most-recent-last, each entry a player-match row with
    `minutes_played` and the PER90_STATS columns. Counting stats are summed and
    divided by the window's total minutes rather than averaged per match, so a
    five-minute cameo with a goal reads as one goal in five minutes of a
    ~900-minute window, not as eighteen goals per 90.
    """
    window = history[-PLAYER_WINDOW:]
    n = len(window)
    if n == 0:
        return {field: None for field in PLAYER_FORM_FIELDS}

    minutes = sum(numeric(h.get("minutes_played")) or 0.0 for h in window)
    ratings = [r for h in window if (r := numeric(h.get("rating"))) is not None]

    form = {
        "rating": sum(ratings) / len(ratings) if ratings else None,
        "minutes_per_appearance": minutes / n,
    }
    for stat in PER90_STATS:
        total = sum(numeric(h.get(stat)) or 0.0 for h in window)
        form[stat] = (total / minutes * 90) if minutes else 0.0
    return form


# --- availability tier -------------------------------------------------------

AVAILABILITY_FIELD_ORDER = (
    "modal_xi_retained",
    "prev_xi_retained",
    "key_players_absent",
    "xi_relative_experience",
    "xi_debutants",
    "xi_mean_minutes_share",
    "days_since_last_match",
)


def modal_xi(recent_xis: list[frozenset[int]]) -> set[int]:
    """The club's settled XI: its most-selected players over recent team sheets.

    Not simply the last XI - a single rotated cup-style lineup would otherwise
    redefine the baseline, and then the return of the first choice team would
    score as *low* continuity. Ties are broken by recency so a newly-established
    starter outranks someone who has just lost his place.
    """
    starts: dict[int, float] = {}
    for age, xi in enumerate(reversed(recent_xis[-MODAL_XI_WINDOW:])):
        for player_id in xi:
            starts[player_id] = starts.get(player_id, 0) + 1 - age * 1e-3
    ranked = sorted(starts, key=lambda p: -starts[p])
    return set(ranked[:11])


class LeagueExperience:
    """Rolling mean appearance count of a league's starters, as of now.

    Raw career appearances cannot be used as a feature directly. Our archive
    starts in 2006, so a 2006 starter with ten seasons behind him reads as a
    debutant and the league-wide mean climbs from ~23 in 2006 to ~120 by 2018
    purely as the observation window fills up. That ramp correlates +0.50 with
    the date, and a model given it will happily learn the era rather than the
    squad - which then evaluates well on a chronological split for entirely
    the wrong reason.

    Dividing by this reference turns the feature into "how experienced is this
    XI compared with the sides it is actually playing against", which is what
    was meant all along and carries no time trend. Rolling rather than
    expanding, so it tracks the ramp instead of lagging behind it.
    """

    WINDOW = 200

    def __init__(self) -> None:
        self._recent: dict[int, list[float]] = {}

    def update(self, league_id: int, xi_mean_appearances: float) -> None:
        recent = self._recent.setdefault(league_id, [])
        recent.append(xi_mean_appearances)
        if len(recent) > self.WINDOW:
            del recent[: -self.WINDOW]

    def reference(self, league_id: int) -> float | None:
        recent = self._recent.get(league_id)
        return sum(recent) / len(recent) if recent else None


def availability_form(
    xi: list[dict],
    recent_xis: list[frozenset[int]],
    season_minutes: dict[int, float],
    appearances: dict[int, int],
    days_since_last_match: float | None,
    league_reference: float | None,
) -> dict:
    """The "who is playing" block, from minutes and team sheets alone.

    `xi` is this fixture's starting eleven (player-match rows, stats not
    needed); `recent_xis` the club's previous team sheets, oldest first;
    `season_minutes` its players' minutes so far this season; `appearances`
    every player's career appearance count in our data, as of now.
    """
    ids = {row["player_id"] for row in xi}
    n = len(ids) or 1

    settled = modal_xi(recent_xis)
    previous = recent_xis[-1] if recent_xis else frozenset()

    # Most-used players by minutes this season. Before a season has any minutes
    # in it (matchday one) nobody qualifies, and the feature correctly reports
    # zero absences rather than eleven.
    key_players = sorted(season_minutes, key=lambda p: -season_minutes[p])[:KEY_PLAYER_COUNT]

    total_minutes = sum(season_minutes.values())
    minutes_share = [
        season_minutes.get(pid, 0.0) / total_minutes if total_minutes else 0.0 for pid in ids
    ]

    mean_appearances = sum(appearances.get(p, 0) for p in ids) / n

    return {
        "modal_xi_retained": len(ids & settled) / 11 if settled else None,
        "prev_xi_retained": len(ids & previous) / 11 if previous else None,
        "key_players_absent": sum(1 for p in key_players if p not in ids),
        "xi_relative_experience": (
            mean_appearances / league_reference if league_reference else None
        ),
        "xi_debutants": sum(1 for p in ids if appearances.get(p, 0) < MIN_APPEARANCES),
        "xi_mean_minutes_share": sum(minutes_share) / n,
        # A team returning from a five-month COVID void or a winter break has
        # form figures averaged over matches that are no longer recent. The
        # model should be able to see that rather than silently average over it
        # (FINDINGS_DATA_AVAILABILITY.md, Ligue 1 2019/20).
        "days_since_last_match": days_since_last_match,
    }


# --- performance tier --------------------------------------------------------

_XI_RATE_STATS = {
    "attack_goals": ("goals",),
    "attack_assists": ("assists",),
    "attack_shots": ("shots_total",),
    "attack_chances": ("key_passes", "chances_created"),
    "defensive_actions": ("tackles", "interceptions", "clearances"),
}

PERFORMANCE_FIELD_ORDER = (
    tuple(f"{group}_rating" for group in POSITION_GROUPS)
    + tuple(f"{group}_count" for group in POSITION_GROUPS)
    + tuple(_XI_RATE_STATS)
    + ("gk_saves", "gk_conceded", "top3_rating", "rating_spread", "imputed_players")
)


def squad_strength(xi: list[dict], forms: dict[int, dict], fallback) -> dict:
    """Aggregate an XI's individual form vectors into a fixed-size block.

    Fixed-size is the whole point: eleven variable-length player vectors cannot
    be fed to a model that expects the same columns every week, so they are
    collapsed by position group. `fallback(row)` supplies a form vector for a
    player without enough history of his own - see `impute_form`. Position
    counts come along for free and encode the shape of the XI (a back three
    versus a back four) without anyone parsing a formation string.
    """
    resolved = []
    imputed = 0
    for row in xi:
        form = forms.get(row["player_id"])
        if form is None or numeric(form.get("rating")) is None:
            form = fallback(row)
            imputed += 1
        resolved.append((row, form))

    block: dict[str, float | None] = {}

    for group in POSITION_GROUPS:
        ratings = [
            form["rating"]
            for row, form in resolved
            if row["position"] == group and numeric(form.get("rating")) is not None
        ]
        block[f"{group}_rating"] = sum(ratings) / len(ratings) if ratings else None
        block[f"{group}_count"] = sum(1 for row, _ in resolved if row["position"] == group)

    # Outfield rates are summed, not averaged: an XI's total expected output is
    # what matters, and summing means swapping a striker for a defender moves
    # the number in the right direction instead of being hidden by the mean.
    for name, stats in _XI_RATE_STATS.items():
        block[name] = sum(
            sum(numeric(form.get(s)) or 0.0 for s in stats) for _, form in resolved
        )

    keepers = [form for row, form in resolved if row["position"] == "goalkeeper"]
    block["gk_saves"] = sum(numeric(f.get("saves")) or 0.0 for f in keepers)
    block["gk_conceded"] = sum(numeric(f.get("goals_conceded")) or 0.0 for f in keepers)

    ratings = sorted(
        (r for _, form in resolved if (r := numeric(form.get("rating"))) is not None),
        reverse=True,
    )
    # Star power and its opposite: two XIs can average the same rating with very
    # different distributions, and a side carrying three outstanding players is
    # not the same proposition as eleven average ones.
    block["top3_rating"] = sum(ratings[:3]) / len(ratings[:3]) if ratings else None
    block["rating_spread"] = _stdev(ratings)
    # How much of the block above is real. The model can use this to discount
    # its own inputs, which matters most for exactly the fixtures where the
    # block is least trustworthy - a promoted side's opening weeks.
    block["imputed_players"] = imputed

    return block


def _stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


class PositionAverages:
    """Running league-and-position mean form, for imputing unknown players.

    Expanding rather than season-scoped, and updated only *after* a fixture's
    features are emitted, so a player's own match can never contribute to the
    average used to impute him. That as-of discipline is the reason this is a
    stateful object walked in date order rather than a groupby over the whole
    table, which would quietly leak the future into every row.
    """

    def __init__(self) -> None:
        self._sums: dict[tuple[int, str], dict[str, float]] = {}
        self._counts: dict[tuple[int, str], int] = {}

    def update(self, league_id: int, position: str | None, form: dict) -> None:
        if numeric(form.get("rating")) is None:
            return
        key = (league_id, position or "unknown")
        sums = self._sums.setdefault(key, dict.fromkeys(PLAYER_FORM_FIELDS, 0.0))
        for field in PLAYER_FORM_FIELDS:
            value = numeric(form.get(field))
            if value is not None:
                sums[field] += value
        self._counts[key] = self._counts.get(key, 0) + 1

    def get(self, league_id: int, position: str | None) -> dict:
        for key in ((league_id, position or "unknown"), (league_id, "unknown")):
            count = self._counts.get(key)
            if count:
                return {f: self._sums[key][f] / count for f in PLAYER_FORM_FIELDS}
        return dict.fromkeys(PLAYER_FORM_FIELDS, None)


def impute_form(averages: PositionAverages, league_id: int):
    """A `fallback` for squad_strength(), closed over the league being played."""

    def fallback(row: dict) -> dict:
        return averages.get(league_id, row.get("position"))

    return fallback
