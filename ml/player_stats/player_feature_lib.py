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

# Club matches over which goal involvements are accumulated. A season's worth,
# and deliberately not season-to-date: in August a season-to-date share is one
# goal out of three and unusable, which is exactly when squads change most.
# Spanning the boundary costs us a departed player keeping his share for a few
# weeks, which is why involvement absence is also reported against the club's
# recent squad rather than against everyone who ever scored in the window.
INVOLVEMENT_WINDOW = 38

# Club matches a player must have appeared in to count as still being at the
# club. Keeps a summer departure, and a long-term injury, from reading as a
# fresh absence every week for months.
SQUAD_RECENCY_WINDOW = 5

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

AVAILABILITY_BASE_FIELDS = (
    "modal_xi_retained",
    "prev_xi_retained",
    "key_players_absent",
    "xi_relative_experience",
    "xi_debutants",
    "xi_mean_minutes_share",
    "days_since_last_match",
)

# The weighted half of the availability block. `key_players_absent` above counts
# heads, which is the aggregation FINDINGS_PHASE9.md put on trial: a club's
# 11th-most-used player missing scores the same as its top scorer missing, and
# spec section 1's claim is specifically about the latter. These two say *how
# much* of the first team is out rather than how many of it.
WEIGHTED_AVAILABILITY_FIELDS = (
    "key_minutes_absent",
    "top_player_absent",
)

# The third importance measure, and the one with variation independent of the
# team sheet. The weighted rework failed because ranking players by season
# minutes and then weighting them by season minutes asks the same question
# twice: key_minutes_absent came out +0.981 correlated with the count it
# replaced. Goal involvements are not how the key players were selected, and
# they concentrate very differently - a squad's minutes are spread over eleven
# near-equal starters, its goals are not.
# The share features - who is missing, weighted by scoring contribution.
INVOLVEMENT_SHARE_FIELDS = (
    "involvement_absent",
    "squad_involvement_absent",
    "top_scorer_absent",
    "xi_involvement_share",
)

# The volume features, which are a different animal entirely and were found by
# accident: how much this club has scored over the window, normalised by its
# league's recent level. These are club-level, not player-level - no lineup
# archive is needed to compute the goals-only one - and they are where the
# measurable gain in this block actually lives. Kept relative rather than raw
# because the raw totals correlate +0.405 with the fixture date (the archive
# fills up), and this project has already been bitten once by shipping a count
# whose level drifts with the observation window.
INVOLVEMENT_VOLUME_FIELDS = (
    "involvement_volume_relative",
    "club_goals_relative",
)

INVOLVEMENT_FIELD_ORDER = INVOLVEMENT_SHARE_FIELDS + INVOLVEMENT_VOLUME_FIELDS

AVAILABILITY_FIELD_ORDER = (
    AVAILABILITY_BASE_FIELDS + WEIGHTED_AVAILABILITY_FIELDS + INVOLVEMENT_FIELD_ORDER
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


class LeagueRollingMean:
    """A league's recent mean of some per-club quantity, as of now.

    Used for two different columns, which is why it is not named after either:
    XI experience and club goal volume. Both are raw counts whose league-wide
    level drifts with the archive rather than with football, and dividing by
    this reference is what turns such a count into a comparison against the
    sides actually being played.

    The case that established the pattern: raw career appearances cannot be
    used as a feature directly. Our archive
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

    def update(self, league_id: int, value: float) -> None:
        recent = self._recent.setdefault(league_id, [])
        recent.append(value)
        if len(recent) > self.WINDOW:
            del recent[: -self.WINDOW]

    def reference(self, league_id: int) -> float | None:
        recent = self._recent.get(league_id)
        return sum(recent) / len(recent) if recent else None


def key_players(season_minutes: dict[int, float]) -> list[int]:
    """The squad's most-used players this season, most-used first.

    Shared by the absence features in both tiers so they always rank the same
    squad the same way; a rating-weighted absence measured against a different
    eleven than the counted one would not be comparable with it.
    """
    return sorted(season_minutes, key=lambda p: -season_minutes[p])[:KEY_PLAYER_COUNT]


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
    key_ids = key_players(season_minutes)

    # Absence weighted by usage. Denominator is the key eleven's own minutes,
    # not the whole squad's, so the feature reads as "what share of the first
    # team's playing time is not on the pitch" and does not shrink just because
    # a club has rotated widely enough to spread minutes over 25 players.
    key_minutes = sum(season_minutes[p] for p in key_ids)
    absent_minutes = sum(season_minutes[p] for p in key_ids if p not in ids)

    total_minutes = sum(season_minutes.values())
    minutes_share = [
        season_minutes.get(pid, 0.0) / total_minutes if total_minutes else 0.0 for pid in ids
    ]

    mean_appearances = sum(appearances.get(p, 0) for p in ids) / n

    return {
        "modal_xi_retained": len(ids & settled) / 11 if settled else None,
        "prev_xi_retained": len(ids & previous) / 11 if previous else None,
        "key_players_absent": sum(1 for p in key_ids if p not in ids),
        "key_minutes_absent": absent_minutes / key_minutes if key_minutes else 0.0,
        # The sharpest form of section 1's claim: not "someone is missing" but
        # "the man who plays every week is missing".
        "top_player_absent": float(bool(key_ids) and key_ids[0] not in ids),
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


class ClubInvolvements:
    """A club's goal involvements over its last INVOLVEMENT_WINDOW matches.

    Involvement is goals + assists, which is the closest thing in this archive
    to "how much of the team's attacking output goes through this player". Held
    per match rather than as a running total so the window can roll; the share
    is recomputed on read, which costs a sum over ~30 players and happens twice
    per fixture.

    A caveat that belongs in the code rather than only in the findings: goals
    are populated from 2005 but assists only from 2008-09, and both undercount
    before 2016 (~2.4 goals per fixture against a true ~2.7). So in the early
    archive this is closer to a goal share than an involvement share. The
    `involvement_sample` feature exposes the window's total so a model can
    discount the thin era rather than being told a confident 0.0.
    """

    def __init__(self) -> None:
        self._matches: dict[int, list[dict[int, float]]] = {}
        self._goals: dict[int, list[float]] = {}

    def update(self, team_id: int, involvements: dict[int, float], goals: float) -> None:
        matches = self._matches.setdefault(team_id, [])
        matches.append(involvements)
        if len(matches) > INVOLVEMENT_WINDOW:
            del matches[:-INVOLVEMENT_WINDOW]

        # Club goals are kept separately from the per-player involvements
        # because the volume they produce turned out to matter far more than
        # the shares, and a goals-only version of it needs no player-level
        # attribution at all - see FINDINGS_PHASE9.md on what that implies.
        scored = self._goals.setdefault(team_id, [])
        scored.append(goals)
        if len(scored) > INVOLVEMENT_WINDOW:
            del scored[:-INVOLVEMENT_WINDOW]

    def totals(self, team_id: int) -> dict[int, float]:
        totals: dict[int, float] = {}
        for match in self._matches.get(team_id, ()):
            for player_id, count in match.items():
                totals[player_id] = totals.get(player_id, 0.0) + count
        return totals

    def volumes(self, team_id: int) -> tuple[float, float]:
        """The club's (involvements, goals) over the window.

        Both are sums over up to INVOLVEMENT_WINDOW matches, so in the thin
        early archive they are lower for reasons that have nothing to do with
        the club. That is what the league reference in `involvement_form` is
        for; the raw figures are never emitted as features.
        """
        return (
            sum(sum(match.values()) for match in self._matches.get(team_id, ())),
            sum(self._goals.get(team_id, ())),
        )

    def shares(self, team_id: int) -> tuple[dict[int, float], float]:
        """Each player's share of the club's window involvements, and the total.

        The total comes back alongside because a share of nothing is not the
        same as a share of nothing much, and the caller has to be able to tell
        0.0-because-nobody-scored from 0.0-because-this-player-did-not.
        """
        totals = self.totals(team_id)
        sample = sum(totals.values())
        if sample <= 0:
            return {}, 0.0
        return {p: c / sample for p, c in totals.items()}, sample


def involvement_form(
    xi: list[dict],
    key_ids: list[int],
    recent_squad: set[int],
    shares: dict[int, float],
    volumes: tuple[float, float],
    references: tuple[float | None, float | None],
) -> dict:
    """Absence weighted by goal involvement rather than by playing time.

    Two absence variants on purpose, because they isolate different things:

    `involvement_absent` keeps the existing key-eleven-by-season-minutes
    definition of who counts as missing and changes *only* the weight, so it is
    a clean swap against `key_players_absent` and `key_minutes_absent`.

    `squad_involvement_absent` drops the minutes-based selection entirely and
    asks how much of the club's recent scoring is not on the pitch, over anyone
    who has played in the last SQUAD_RECENCY_WINDOW matches. This is the version
    with variation independent of the team sheet - a rotated striker with a
    third of the club's goals is invisible to a minutes ranking and dominant
    here - and therefore the one that actually tests spec section 1.

    `recent_squad` is every player who has appeared for the club recently, which
    is what keeps a summer departure or a long-term injury from being counted as
    a fresh absence indefinitely.

    `volumes` and `references` carry the club's (involvements, goals) over the
    window and its league's recent mean of each. The two volume features are the
    ones that measurably help, and they are not really player features - see the
    note on INVOLVEMENT_VOLUME_FIELDS.
    """
    ids = {row["player_id"] for row in xi}

    ranked = sorted(shares, key=lambda p: -shares[p])
    top_scorer = next((p for p in ranked if p in recent_squad), None)

    return {
        "involvement_absent": sum(shares.get(p, 0.0) for p in key_ids if p not in ids),
        "squad_involvement_absent": sum(
            share for p, share in shares.items() if p in recent_squad and p not in ids
        ),
        "top_scorer_absent": float(top_scorer is not None and top_scorer not in ids),
        "xi_involvement_share": sum(shares.get(p, 0.0) for p in ids),
        "involvement_volume_relative": _relative(volumes[0], references[0]),
        "club_goals_relative": _relative(volumes[1], references[1]),
    }


def _relative(value: float, reference: float | None) -> float | None:
    """A count against its league's recent level, or None before one exists."""
    return value / reference if reference else None


# --- performance tier --------------------------------------------------------

_XI_RATE_STATS = {
    "attack_goals": ("goals",),
    "attack_assists": ("assists",),
    "attack_shots": ("shots_total",),
    "attack_chances": ("key_passes", "chances_created"),
    "defensive_actions": ("tackles", "interceptions", "clearances"),
}

PERFORMANCE_BASE_FIELDS = (
    tuple(f"{group}_rating" for group in POSITION_GROUPS)
    + tuple(f"{group}_count" for group in POSITION_GROUPS)
    + tuple(_XI_RATE_STATS)
    + ("gk_saves", "gk_conceded", "top3_rating", "rating_spread", "imputed_players")
)

# The weighted half of the performance block, and the reason this rework exists.
# Everything above treats an XI as eleven interchangeable slots: a per-position
# mean gives an academy debutant the same vote as the club captain. The only
# base feature that ranked at all was top3_rating - a distribution statistic -
# which is what pointed at the aggregation rather than at the input
# (FINDINGS_PHASE9.md, open item 2).
#
# These weight each player by how much this club actually plays him, and extend
# the same weighting to who is missing.
# Emitted by squad_strength() ...
WEIGHTED_SQUAD_FIELDS = ("xi_weighted_rating", "rating_top1", "rating_bottom3")
# ... and by absence_impact(), which needs the forms of players who are *not* in
# the XI and so cannot come out of the same function.
ABSENCE_FIELD_ORDER = ("absent_rating_mean", "absent_rating_weighted")

WEIGHTED_PERFORMANCE_FIELDS = WEIGHTED_SQUAD_FIELDS + ABSENCE_FIELD_ORDER

SQUAD_FIELD_ORDER = PERFORMANCE_BASE_FIELDS + WEIGHTED_SQUAD_FIELDS
PERFORMANCE_FIELD_ORDER = SQUAD_FIELD_ORDER + ABSENCE_FIELD_ORDER


def squad_strength(
    xi: list[dict], forms: dict[int, dict], fallback, weights: dict[int, float] | None = None
) -> dict:
    """Aggregate an XI's individual form vectors into a fixed-size block.

    Fixed-size is the whole point: eleven variable-length player vectors cannot
    be fed to a model that expects the same columns every week, so they are
    collapsed by position group. `fallback(row)` supplies a form vector for a
    player without enough history of his own - see `impute_form`. Position
    counts come along for free and encode the shape of the XI (a back three
    versus a back four) without anyone parsing a formation string.

    `weights` is each player's minutes so far this season, used for the weighted
    rating. Unweighted means are what this block was criticised for: a squad's
    quality is not the average of eleven equal votes when three of the eleven
    are reserves covering for injuries. Absent or all-zero weights degrade to a
    plain mean, which is the correct reading on matchday one.
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
    # The rest of the distribution that top3_rating only half-describes: the
    # single best player, and the weakest three - a side is often beaten
    # through the three places it had to patch, not through its stars.
    block["rating_top1"] = ratings[0] if ratings else None
    block["rating_bottom3"] = sum(ratings[-3:]) / len(ratings[-3:]) if ratings else None
    block["xi_weighted_rating"] = weighted_mean(
        [
            (r, (weights or {}).get(row["player_id"], 0.0))
            for row, form in resolved
            if (r := numeric(form.get("rating"))) is not None
        ]
    )
    # How much of the block above is real. The model can use this to discount
    # its own inputs, which matters most for exactly the fixtures where the
    # block is least trustworthy - a promoted side's opening weeks.
    block["imputed_players"] = imputed

    return block


def weighted_mean(pairs: list[tuple[float, float]]) -> float | None:
    """Mean of (value, weight) pairs, degrading to a plain mean without weights.

    The degradation matters more than the formula: every club's first fixture
    of a season has no minutes to weight by, and a feature that went null there
    would be missing exactly where squad churn is largest.
    """
    if not pairs:
        return None
    total = sum(w for _, w in pairs)
    if total <= 0:
        return sum(v for v, _ in pairs) / len(pairs)
    return sum(v * w for v, w in pairs) / total


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


def absence_impact(
    absent: list[dict],
    season_minutes: dict[int, float],
    forms: dict[int, dict],
    fallback,
) -> dict:
    """How good the missing key players are, not just how many there are.

    This is the half of the rework that tests spec section 1 directly. The
    counted version cannot tell a club's top scorer from its eleventh-most-used
    squad player; here each absentee carries his own rolling rating.

    `absent` is one row per missing key player (`player_id` and `position`, the
    latter only so an unknown player can be imputed from his position group),
    and `season_minutes` the whole squad's minutes, which supply the weight.

    Two features rather than one because they answer different questions and a
    single product confounds them: the mean says *how good* the missing players
    are, the weighted sum says *how much* of the side is missing and how good
    it was. A fringe international out injured moves the first and not the
    second; three regulars suspended move the second and not the first.
    """
    total_minutes = sum(season_minutes.values())
    ratings: list[float] = []
    weighted = 0.0
    for row in absent:
        form = forms.get(row["player_id"])
        if form is None or numeric(form.get("rating")) is None:
            form = fallback(row)
        rating = numeric(form.get("rating"))
        if rating is None:
            continue
        ratings.append(rating)
        share = season_minutes.get(row["player_id"], 0.0) / total_minutes if total_minutes else 0.0
        weighted += rating * share

    return {
        "absent_rating_mean": sum(ratings) / len(ratings) if ratings else None,
        "absent_rating_weighted": weighted,
    }
