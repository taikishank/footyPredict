"""Opponent-strength ratings, shared between the offline experiment
(build_rating_features.py) and - if one of these arms is promoted - the
FastAPI inference service, the same way feature_lib.py is shared today.

The incumbent's team_form() is opponent-blind: eight matches against
relegation sides and eight against the top four produce the same vector.
Every rating here fixes that by being *sequential* - a team's rating is a
running summary of every result it has produced, and each update is scaled by
how good the opponent was at the time.

Three arms, one interface (Rating below), because they differ only in the
update rule and the columns they emit:

  EloRating        W/D/L only. The textbook version; fewest knobs.
  GoalDiffElo      the same, with the margin of victory folded into the update.
  AttackDefence    two ratings per team, fitted on the Poisson likelihood of
                   the scoreline. Also the parameterisation a Dixon-Coles
                   goals model needs, so this arm doubles as that foundation.

Two facts about the archive drive the shared machinery (see the header of
build_rating_features.py for how they were established):

  Five leagues, zero cross-league fixtures, zero teams in more than one
  league. The rating pool is five disconnected components, so every mean and
  every regression target below is *per-league*. A global one would average
  over teams that can never play each other.

  116 of 211 teams have a gap season. A team relegated in 2015 and promoted
  in 2019 would otherwise return carrying its 2015 rating - stale, and wrong
  in a known direction. season_boundary() regresses toward the league mean
  once per season *elapsed*, so four years away costs four regressions.
"""
from __future__ import annotations

import math

# Elo points of home advantage, applied to the home side's rating inside the
# expectation but never stored. 65 is the conventional football value and is
# left fixed rather than tuned: it trades off directly against K, and letting
# a sweep move both invites the two to chase each other.
HOME_ADVANTAGE = 65.0

# Every team starts here, and league means are anchored here at the start.
INIT_RATING = 1500.0

# New teams enter below the league mean because promoted sides are
# systematically weaker than the ones they replaced - seeding them *at* the
# mean hands a free over-rating to every club that plays them in August.
# Measured rather than assumed: see build_rating_features.py --diagnose.
PROMOTED_OFFSET = 60.0

# Fraction of the way a rating is pulled back to its league mean at each
# season boundary, for squad turnover. Applied once per season elapsed, so a
# club out of the division for four years is regressed four times and ends up
# essentially at the mean, which is the right prior for a side nobody in this
# archive has seen for four years.
SEASON_REGRESSION = 0.25


class LeagueMeans:
    """Per-league mean rating over the clubs *currently* in that league.

    Elo is zero-sum only in a closed pool. This one is not: clubs are
    relegated out and promoted in every summer, so a fixed anchor would
    slowly become the wrong regression target.

    The membership therefore has to be the current season's cohort, not every
    club ever seen. Averaging over the accumulated roster instead leaves the
    level columns correlated +0.21 with the fixture date, because the
    reference set grows for 26 seasons while the live league stays at twenty
    - the same era-proxy failure recenter() exists to prevent, arriving by a
    different route.

    `reference` is the cohort that most recently finished a season, which is
    the best available estimate of "a typical club in this league" at the
    August rollover, when every seeding and regression decision is made.
    """

    def __init__(self) -> None:
        self._current: dict[int, set[int]] = {}
        self._previous: dict[int, set[int]] = {}

    def note(self, league_id: int, team_id: int) -> None:
        self._current.setdefault(league_id, set()).add(team_id)

    def roll_season(self, league_id: int) -> None:
        """Retire the finished season's cohort into the reference slot."""
        finished = self._current.get(league_id)
        if finished:
            self._previous[league_id] = finished
        self._current[league_id] = set()

    def reference(self, league_id: int) -> set[int]:
        # Falls back to the part-filled current cohort during a league's
        # founding season, before any rollover has happened.
        return self._previous.get(league_id) or self._current.get(league_id) or set()

    def has_members(self, league_id: int) -> bool:
        return bool(self.reference(league_id))

    def mean(self, league_id: int, ratings: dict[int, float]) -> float:
        members = self.reference(league_id)
        if not members:
            # Nothing has played in this league yet, so there are no
            # incumbents to be promoted *into* - the founding cohort all start
            # level at the anchor, and PROMOTED_OFFSET only applies to clubs
            # arriving later. `is_founding` below is what suppresses it.
            return INIT_RATING
        return sum(ratings.get(t, INIT_RATING) for t in members) / len(members)


def _expected(rating_home: float, rating_away: float) -> float:
    """Standard Elo expectation for the home side, home advantage included.

    This is the *score* expectation on a 0-1 scale where a draw is 0.5, not a
    win probability - with three outcomes those are different quantities, and
    conflating them is the classic way to misread an Elo column. The model
    downstream is what turns a rating gap into P(home)/P(draw)/P(away); this
    only has to be a monotone, well-scaled summary of the gap.
    """
    return 1.0 / (1.0 + 10.0 ** (-(rating_home + HOME_ADVANTAGE - rating_away) / 400.0))


class Rating:
    """Walked in date order: features() strictly before update() for each
    fixture, so a row never sees its own result.

    Callers must also drive season_boundary() when a club's season changes,
    which is what makes the gap-season regression happen.
    """

    #: Column names features() returns, in a fixed order, so a caller building
    #: a vector by hand matches the training column order (the same contract
    #: feature_lib.FORM_FIELD_ORDER carries).
    FIELD_ORDER: tuple[str, ...] = ()

    def features(self, home_id: int, away_id: int, league_id: int) -> dict[str, float]:
        raise NotImplementedError

    def update(
        self, home_id: int, away_id: int, league_id: int, home_goals: int, away_goals: int
    ) -> None:
        raise NotImplementedError

    def season_boundary(self, league_id: int, team_id: int, seasons_elapsed: int) -> None:
        raise NotImplementedError

    def recenter(self, league_id: int) -> None:
        """Shift a league's live ratings so their mean returns to the anchor.

        Required, not cosmetic. Only rating *differences* affect a prediction,
        so this is a no-op for the model - but without it the pool mean bleeds
        downward, because every August injects clubs seeded below the mean
        (PROMOTED_OFFSET) while the zero-sum update never removes that deficit
        again. Left alone, the level columns correlate -0.42 with the fixture
        date and become an era proxy: the exact failure Phase 9 recorded for
        career appearance counts, where a column that tracks the archive
        rather than the football scores well on a chronological split for
        entirely the wrong reason.
        """
        raise NotImplementedError


class EloRating(Rating):
    """Plain Elo: the update sees win, draw or loss and nothing else.

    K is the only real knob. 20 is the conventional club-football value; it
    sets how many matches of memory the rating has (roughly 1/K of the gap is
    closed per match), and is swept in build_rating_features.py rather than
    assumed.
    """

    FIELD_ORDER = ("home_elo", "away_elo", "elo_diff", "elo_expected_home")

    def __init__(self, k: float = 20.0) -> None:
        self.k = k
        self.ratings: dict[int, float] = {}
        self.means = LeagueMeans()

    def _get(self, team_id: int, league_id: int) -> float:
        """A club's rating, seeding it below the league mean on first sight."""
        if team_id not in self.ratings:
            mean = self.means.mean(league_id, self.ratings)
            is_founding = not self.means.has_members(league_id)
            self.ratings[team_id] = mean if is_founding else mean - PROMOTED_OFFSET
        return self.ratings[team_id]

    def features(self, home_id: int, away_id: int, league_id: int) -> dict[str, float]:
        h = self._get(home_id, league_id)
        a = self._get(away_id, league_id)
        # The explicit difference is not redundant with the two levels: a
        # gradient-boosted tree splits on one column at a time and cannot
        # construct (h - a) for itself, so handing it the gap is what lets it
        # use the rating at all. The levels stay because "1600 vs 1500" and
        # "1500 vs 1400" are the same gap in different leagues.
        return {
            "home_elo": h,
            "away_elo": a,
            "elo_diff": h - a,
            "elo_expected_home": _expected(h, a),
        }

    def _score(self, home_goals: int, away_goals: int) -> float:
        if home_goals > away_goals:
            return 1.0
        return 0.5 if home_goals == away_goals else 0.0

    def _multiplier(self, home_goals: int, away_goals: int) -> float:
        return 1.0

    def update(
        self, home_id: int, away_id: int, league_id: int, home_goals: int, away_goals: int
    ) -> None:
        h = self._get(home_id, league_id)
        a = self._get(away_id, league_id)
        self.means.note(league_id, home_id)
        self.means.note(league_id, away_id)

        delta = (
            self.k
            * self._multiplier(home_goals, away_goals)
            * (self._score(home_goals, away_goals) - _expected(h, a))
        )
        # Zero-sum by construction: what the home side gains the away side
        # loses. This is what makes the rating a measure of *relative*
        # strength and keeps the league mean stable between transfer windows.
        self.ratings[home_id] = h + delta
        self.ratings[away_id] = a - delta

    def season_boundary(self, league_id: int, team_id: int, seasons_elapsed: int) -> None:
        if team_id not in self.ratings or seasons_elapsed <= 0:
            return
        target = self.means.mean(league_id, self.ratings)
        keep = (1.0 - SEASON_REGRESSION) ** seasons_elapsed
        self.ratings[team_id] = target + (self.ratings[team_id] - target) * keep

    def recenter(self, league_id: int) -> None:
        self.means.roll_season(league_id)
        members = self.means.reference(league_id)
        if not members:
            return
        shift = INIT_RATING - self.means.mean(league_id, self.ratings)
        for team_id in members:
            self.ratings[team_id] = self.ratings.get(team_id, INIT_RATING) + shift


class GoalDiffElo(EloRating):
    """Elo with the margin of victory folded into the update.

    A 4-0 and a 1-0 are the same evidence to plain Elo, which is a real loss
    of information given the scoreline is already in the archive. The
    multiplier is the World Football Elo formulation: 1 for a one-goal margin,
    1.5 for two, and (11 + m) / 8 beyond that - deliberately concave, because
    the difference between winning by four and by five says much less about a
    side than the difference between winning by one and by two.
    """

    FIELD_ORDER = ("home_gdelo", "away_gdelo", "gdelo_diff", "gdelo_expected_home")

    def features(self, home_id: int, away_id: int, league_id: int) -> dict[str, float]:
        # Same four quantities as the parent, under this arm's own names, so
        # both arms' columns can sit in one table without colliding.
        base = super().features(home_id, away_id, league_id)
        return dict(zip(self.FIELD_ORDER, (base[k] for k in EloRating.FIELD_ORDER)))

    def _multiplier(self, home_goals: int, away_goals: int) -> float:
        margin = abs(home_goals - away_goals)
        if margin <= 1:
            return 1.0
        if margin == 2:
            return 1.5
        return (11.0 + margin) / 8.0


class AttackDefence(Rating):
    """Two ratings per club, fitted online on the Poisson likelihood of the
    scoreline:

        goals_home ~ Poisson(exp(mu + atk_home - def_away + hfa))
        goals_away ~ Poisson(exp(mu + atk_away - def_home))

    Richer than a single scalar - a side that wins 3-2 every week and one that
    wins 1-0 every week have the same Elo and very different ratings here -
    and it is exactly the parameterisation Dixon-Coles needs, so promoting
    this arm would also be the groundwork for a goals model that derives 1X2
    from a score distribution rather than classifying outcomes directly.

    Fitted by online gradient ascent rather than a batch solve, because the
    whole point is a *pre-match* value for every fixture in date order: the
    gradient of the Poisson log-likelihood with respect to atk_home is simply
    (goals_home - lambda_home), which is the same "surprise drives the update"
    shape Elo has.
    """

    FIELD_ORDER = (
        "home_atk", "home_def", "away_atk", "away_def",
        "exp_goals_home", "exp_goals_away", "exp_goal_diff",
    )

    # Smaller than it looks: this is a multiplicative-scale step, so 0.02 moves
    # a rating by ~2% of a goal per unit of surprise.
    LEARNING_RATE = 0.02
    # Ratings are log-scale multipliers on a goal rate; without a clamp a club
    # on a freak run can run away exponentially before the pull-back arrives.
    MAX_RATING = 1.5
    # Guards exp() against the same runaway when computing a rate.
    MAX_LAMBDA = 8.0

    def __init__(self, learning_rate: float | None = None) -> None:
        self.lr = self.LEARNING_RATE if learning_rate is None else learning_rate
        self.atk: dict[int, float] = {}
        self.dfn: dict[int, float] = {}
        # Global log baseline goal rate and home advantage, both learned
        # online from the same gradient - they are league-invariant enough
        # not to be worth a per-league parameter each, and learning them
        # beats hard-coding a scoring level that drifts across 26 seasons.
        # Per-league rather than global. The five leagues never meet, so
        # there is no pooling benefit, and their scoring levels genuinely
        # differ - but the binding reason is recenter(): it can only stay
        # exactly neutral by moving the baseline to absorb what it takes out
        # of the ratings, and a global baseline would push that correction
        # onto the other four leagues.
        self.mu: dict[int, float] = {}
        self.hfa: dict[int, float] = {}
        # Same cohort bookkeeping as the Elo arms, for the same reason.
        self.means = LeagueMeans()

    def _seed(self, team_id: int, league_id: int) -> None:
        if team_id in self.atk:
            return
        # Promoted sides score less and concede more; one offset, applied with
        # the sign each rating needs, keeps the asymmetry the Elo arm's single
        # PROMOTED_OFFSET expresses.
        members = self.means.reference(league_id)
        if members:
            mean_atk = sum(self.atk.get(t, 0.0) for t in members) / len(members)
            mean_def = sum(self.dfn.get(t, 0.0) for t in members) / len(members)
        else:
            mean_atk = mean_def = 0.0
        self.atk[team_id] = mean_atk - 0.10
        self.dfn[team_id] = mean_def - 0.10

    #: Starting log goal rate, about 1.35 goals a side - close enough to the
    #: archive's average (1.53 home, 1.18 away) that the online fit only
    #: trims it, settling at ~0.285 across all five leagues.
    INIT_MU = math.log(1.35)

    #: Starting log home advantage. Not zero, and this matters: fitted from
    #: zero it climbs to ~0.20 over the archive, which makes every early
    #: fixture's exp_goals_home systematically smaller than every late one and
    #: leaves that column correlated +0.19 with the fixture date - a burn-in
    #: ramp indistinguishable, to a tree on a chronological split, from the
    #: career appearance bug in FINDINGS_PHASE9.md. Seeded at the level the
    #: fit settles on instead, which puts it on the same footing as the Elo
    #: arms' fixed HOME_ADVANTAGE rather than being a fitted quantity.
    INIT_HFA = 0.20

    def _rates(self, home_id: int, away_id: int, league_id: int) -> tuple[float, float]:
        mu = self.mu.setdefault(league_id, self.INIT_MU)
        hfa = self.hfa.setdefault(league_id, self.INIT_HFA)
        cap = math.log(self.MAX_LAMBDA)
        lam_h = math.exp(min(mu + self.atk[home_id] - self.dfn[away_id] + hfa, cap))
        lam_a = math.exp(min(mu + self.atk[away_id] - self.dfn[home_id], cap))
        return lam_h, lam_a

    def features(self, home_id: int, away_id: int, league_id: int) -> dict[str, float]:
        self._seed(home_id, league_id)
        self._seed(away_id, league_id)
        lam_h, lam_a = self._rates(home_id, away_id, league_id)
        return {
            "home_atk": self.atk[home_id],
            "home_def": self.dfn[home_id],
            "away_atk": self.atk[away_id],
            "away_def": self.dfn[away_id],
            "exp_goals_home": lam_h,
            "exp_goals_away": lam_a,
            # The tree's equivalent of elo_diff: the one number that collapses
            # four ratings into "who is expected to win, and by how much".
            "exp_goal_diff": lam_h - lam_a,
        }

    def _clamp(self, value: float) -> float:
        return max(-self.MAX_RATING, min(self.MAX_RATING, value))

    def update(
        self, home_id: int, away_id: int, league_id: int, home_goals: int, away_goals: int
    ) -> None:
        self._seed(home_id, league_id)
        self._seed(away_id, league_id)
        self.means.note(league_id, home_id)
        self.means.note(league_id, away_id)

        lam_h, lam_a = self._rates(home_id, away_id, league_id)
        err_h = home_goals - lam_h
        err_a = away_goals - lam_a

        # d/d(atk_home) log L = goals_home - lambda_home, and defence enters
        # the opponent's rate with the opposite sign - so conceding fewer than
        # expected raises a defence rating.
        self.atk[home_id] = self._clamp(self.atk[home_id] + self.lr * err_h)
        self.dfn[away_id] = self._clamp(self.dfn[away_id] - self.lr * err_h)
        self.atk[away_id] = self._clamp(self.atk[away_id] + self.lr * err_a)
        self.dfn[home_id] = self._clamp(self.dfn[home_id] - self.lr * err_a)

        # The two league-level terms move far more slowly than a club rating:
        # they are averaging over every fixture in the league, not over one
        # club's.
        self.mu.setdefault(league_id, self.INIT_MU)
        self.hfa.setdefault(league_id, self.INIT_HFA)
        self.mu[league_id] += self.lr * 0.01 * (err_h + err_a)
        self.hfa[league_id] += self.lr * 0.01 * (err_h - err_a)

    def season_boundary(self, league_id: int, team_id: int, seasons_elapsed: int) -> None:
        if team_id not in self.atk or seasons_elapsed <= 0:
            return
        members = self.means.reference(league_id)
        if members:
            target_atk = sum(self.atk.get(t, 0.0) for t in members) / len(members)
            target_dfn = sum(self.dfn.get(t, 0.0) for t in members) / len(members)
        else:
            target_atk = target_dfn = 0.0
        keep = (1.0 - SEASON_REGRESSION) ** seasons_elapsed
        self.atk[team_id] = target_atk + (self.atk[team_id] - target_atk) * keep
        self.dfn[team_id] = target_dfn + (self.dfn[team_id] - target_dfn) * keep

    def recenter(self, league_id: int) -> None:
        # Anchored at 0 rather than INIT_RATING: these are log-scale offsets
        # against the global baseline rate `mu`, so a zero-mean league is the
        # one whose scoring `mu` already describes.
        self.means.roll_season(league_id)
        members = self.means.reference(league_id)
        if not members:
            return
        shifts = []
        for ratings in (self.atk, self.dfn):
            shift = -sum(ratings.get(t, 0.0) for t in members) / len(members)
            for team_id in members:
                ratings[team_id] = ratings.get(team_id, 0.0) + shift
            shifts.append(shift)

        # Attack and defence enter a rate with opposite signs and against
        # *different* clubs, so unlike Elo's single scalar they cannot both be
        # zeroed by a shift that leaves the rates alone. Moving the league
        # baseline by the difference is what restores exactness: every
        # exp_goals value this arm emits is unchanged, while both rating
        # families are re-centred on zero and stop tracking the archive.
        attack_shift, defence_shift = shifts
        self.mu[league_id] = (
            self.mu.setdefault(league_id, self.INIT_MU) - attack_shift + defence_shift
        )


#: Arm name -> factory. build_rating_features.py emits every arm's columns in
#: one pass so the arms are compared on byte-identical rows.
ARMS = {
    "elo": EloRating,
    "gdelo": GoalDiffElo,
    "atkdef": AttackDefence,
}
