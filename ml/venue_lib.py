"""Home advantage as a rating parameter, and the club-venue arms it displaces.

FINDINGS_MARKET.md put away wins at 61.8% of the gap to the closing line and
named club-level venue strength as the fix: "a side that travels badly and one
that doesn't produce the same vector". The pre-ablation probe in
build_venue_features.py --diagnose says that side does not exist - a club's
venue gap, measured on half its matches, does not predict its own other half
(split-half r = -0.03). The club-to-club spread in home advantage is sampling
noise end to end.

What the same probe *did* find is that league home advantage has fallen -
log HFA 0.303 over 2005-2015 against 0.201 over 2021-2025, every league moving
the same way - while AttackDefence carries a nearly frozen one. Its hfa is
seeded at INIT_HFA = 0.20 and moves at `lr * HFA_RATE` = 0.0002 per fixture
against a league-wide average, so the served exp_goals_home overstates the
home side across the whole validation era, in exactly the direction the
market's outcome split shows.

So the arms here split into two families that answer different claims:

  AdaptiveHFA    the same rating, with the hfa update rate raised. The
                 cheapest possible version of "the parameter is stale".

  RollingHFA     hfa read off the league's last WINDOW fixtures rather than
                 learned by gradient at all. A window has a horizon a
                 learning rate only approximates, and 2020 is the case that
                 separates them: empty stadiums moved HFA to 0.119 for one
                 season and back, which a window tracks and un-tracks on a
                 known timescale.

  VenueAttackDefence   four ratings per club - attack and defence, home and
                 away - each updated only at the venue it describes. The arm
                 FINDINGS_MARKET actually asked for. Predicted null by the
                 probe; measured anyway, because the hypothesis was ranked
                 first and a probe is not an ablation.

  PlaceboVenue   VenueAttackDefence's exact shape with the two slots assigned
                 by fixture id rather than by venue. Same parameter count,
                 same halving of the evidence behind each rating, no football
                 in the split. FINDINGS_LEAGUE.md is the precedent: `interact`
                 cleared its interval and was beaten by a random partition of
                 the same shape, and this phase is not going to promote a
                 second one of those.

Note what is *not* an arm here: handing the trees a recent-home-rate column.
FINDINGS_LEAGUE.md already measured that as `character` and it was worth
-0.00005. Telling the model the current HFA does not help while the rating
column it leans on has a stale one baked into it multiplicatively - a
depth-4 tree cannot unpick that with a separate additive column. Every arm
here changes the rating itself.
"""
from __future__ import annotations

import math
from collections import deque

from ratings_lib import AttackDefence

# Fixtures of league history behind RollingHFA's estimate. 380 is one season
# of a 20-club league, matching league_lib.WINDOW: short enough to have moved
# through 2020's empty stadiums, long enough that the estimate's own noise
# (~0.03 on the log scale) is well under the 0.10 drift it has to track.
HFA_WINDOW = 380

# Below this the window's estimate is noisier than the drift it describes, so
# the arm falls back to AttackDefence's seed rather than reading a number that
# means nothing.
HFA_MIN_FIXTURES = 200


class AdaptiveHFA(AttackDefence):
    """AttackDefence with the home-advantage update rate exposed.

    `hfa_rate` multiplies the shared learning rate, exactly as the served
    class does with its hard-coded 0.01 - so hfa_rate=0.01 reproduces the
    served ratings column for column, which build_venue_features.py asserts
    rather than trusts.
    """

    #: The served rate, kept as the default so this class is a strict
    #: generalisation of its parent.
    HFA_RATE = 0.01

    def __init__(self, learning_rate: float | None = None,
                 hfa_rate: float | None = None) -> None:
        super().__init__(learning_rate)
        self.hfa_rate = self.HFA_RATE if hfa_rate is None else hfa_rate

    def features(self, home_id: int, away_id: int, league_id: int,
                 fixture_id: int | None = None) -> dict[str, float]:
        return super().features(home_id, away_id, league_id)

    def update(self, home_id: int, away_id: int, league_id: int,
               home_goals: int, away_goals: int,
               fixture_id: int | None = None) -> None:
        # Reproduces the parent update with the hfa step scaled. Written out
        # rather than delegated because the parent applies its own 0.01 inline
        # and there is no hook between the two halves of it.
        self._seed(home_id, league_id)
        self._seed(away_id, league_id)
        self.means.note(league_id, home_id)
        self.means.note(league_id, away_id)

        lam_h, lam_a = self._rates(home_id, away_id, league_id)
        err_h = home_goals - lam_h
        err_a = away_goals - lam_a

        self.atk[home_id] = self._clamp(self.atk[home_id] + self.lr * err_h)
        self.dfn[away_id] = self._clamp(self.dfn[away_id] - self.lr * err_h)
        self.atk[away_id] = self._clamp(self.atk[away_id] + self.lr * err_a)
        self.dfn[home_id] = self._clamp(self.dfn[home_id] - self.lr * err_a)

        self.mu.setdefault(league_id, self.INIT_MU)
        self.hfa.setdefault(league_id, self.INIT_HFA)
        self.mu[league_id] += self.lr * 0.01 * (err_h + err_a)
        self.hfa[league_id] += self.lr * self.hfa_rate * (err_h - err_a)

    def season_boundary(self, league_id: int, team_id: int, seasons_elapsed: int) -> None:
        super().season_boundary(league_id, team_id, seasons_elapsed)


class RollingHFA(AdaptiveHFA):
    """Home advantage read off the league's recent fixtures, not learned.

    The estimator is the log ratio of mean home to mean away goals over the
    window, which is what `hfa` means in the parent's rate: with atk and dfn
    re-centred on zero per league, E[log lam_h] - E[log lam_a] is the hfa
    term and nothing else.

    Causal by construction - the deque is appended in update(), strictly after
    features() has read it, the same order every rating in this project keeps.
    """

    def __init__(self, learning_rate: float | None = None,
                 window: int = HFA_WINDOW, min_fixtures: int = HFA_MIN_FIXTURES) -> None:
        # hfa_rate=0 because this arm's hfa is not a fitted parameter at all;
        # leaving the gradient on would have the window and the gradient
        # fighting over the same term.
        super().__init__(learning_rate, hfa_rate=0.0)
        self.window = window
        self.min_fixtures = min_fixtures
        self._recent: dict[int, deque[tuple[int, int]]] = {}

    def _rolling_hfa(self, league_id: int) -> float:
        recent = self._recent.get(league_id)
        if recent is None or len(recent) < self.min_fixtures:
            return self.INIT_HFA
        home = sum(h for h, _ in recent) / len(recent)
        away = sum(a for _, a in recent) / len(recent)
        if home <= 0.0 or away <= 0.0:
            return self.INIT_HFA
        return math.log(home) - math.log(away)

    def _rates(self, home_id: int, away_id: int, league_id: int) -> tuple[float, float]:
        # The window's estimate replaces the stored parameter for this
        # fixture. Written into self.hfa rather than passed down so that
        # recenter()'s mu bookkeeping, which never touches hfa, stays valid.
        self.hfa[league_id] = self._rolling_hfa(league_id)
        return super()._rates(home_id, away_id, league_id)

    def update(self, home_id: int, away_id: int, league_id: int,
               home_goals: int, away_goals: int,
               fixture_id: int | None = None) -> None:
        super().update(home_id, away_id, league_id, home_goals, away_goals, fixture_id)
        recent = self._recent.get(league_id)
        if recent is None:
            recent = self._recent[league_id] = deque(maxlen=self.window)
        recent.append((home_goals, away_goals))


class VenueAttackDefence(AttackDefence):
    """Four ratings per club: attack and defence, home and away.

    A club's home ratings are updated only by its home matches and read only
    when it is the home side, so "travels badly" has somewhere to live. The
    cost is that each rating sees half the evidence, which is why PlaceboVenue
    exists: halving the evidence is a change in its own right, and its effect
    has to be measured separately from the venue split's.
    """

    #: Slot names, purely for readability of the (team, slot) keys.
    HOME, AWAY = "H", "A"

    def _slots(self, fixture_id: int | None) -> tuple[str, str]:
        """Which slot each side reads and writes. Venue, here; random in the
        placebo below."""
        return self.HOME, self.AWAY

    def _seed_slots(self, team_id: int, league_id: int) -> None:
        if (team_id, self.HOME) in self.atk:
            return
        members = self.means.reference(league_id)
        if members:
            # Averaged over both slots of each member, so the seed is a club's
            # overall level and not one venue's - a promoted side has no
            # reason to arrive with an asymmetry.
            vals = [(self.atk.get((t, s), 0.0), self.dfn.get((t, s), 0.0))
                    for t in members for s in (self.HOME, self.AWAY)]
            mean_atk = sum(a for a, _ in vals) / len(vals)
            mean_def = sum(d for _, d in vals) / len(vals)
        else:
            mean_atk = mean_def = 0.0
        for slot in (self.HOME, self.AWAY):
            self.atk[(team_id, slot)] = mean_atk - 0.10
            self.dfn[(team_id, slot)] = mean_def - 0.10

    def _venue_rates(self, home_key, away_key, league_id: int) -> tuple[float, float]:
        mu = self.mu.setdefault(league_id, self.INIT_MU)
        hfa = self.hfa.setdefault(league_id, self.INIT_HFA)
        cap = math.log(self.MAX_LAMBDA)
        lam_h = math.exp(min(mu + self.atk[home_key] - self.dfn[away_key] + hfa, cap))
        lam_a = math.exp(min(mu + self.atk[away_key] - self.dfn[home_key], cap))
        return lam_h, lam_a

    def features(self, home_id: int, away_id: int, league_id: int,
                 fixture_id: int | None = None) -> dict[str, float]:
        self._seed_slots(home_id, league_id)
        self._seed_slots(away_id, league_id)
        hs, as_ = self._slots(fixture_id)
        home_key, away_key = (home_id, hs), (away_id, as_)
        lam_h, lam_a = self._venue_rates(home_key, away_key, league_id)
        return {
            "home_atk": self.atk[home_key],
            "home_def": self.dfn[home_key],
            "away_atk": self.atk[away_key],
            "away_def": self.dfn[away_key],
            "exp_goals_home": lam_h,
            "exp_goals_away": lam_a,
            "exp_goal_diff": lam_h - lam_a,
        }

    def update(self, home_id: int, away_id: int, league_id: int,
               home_goals: int, away_goals: int,
               fixture_id: int | None = None) -> None:
        self._seed_slots(home_id, league_id)
        self._seed_slots(away_id, league_id)
        self.means.note(league_id, home_id)
        self.means.note(league_id, away_id)

        hs, as_ = self._slots(fixture_id)
        home_key, away_key = (home_id, hs), (away_id, as_)
        lam_h, lam_a = self._venue_rates(home_key, away_key, league_id)
        err_h = home_goals - lam_h
        err_a = away_goals - lam_a

        self.atk[home_key] = self._clamp(self.atk[home_key] + self.lr * err_h)
        self.dfn[away_key] = self._clamp(self.dfn[away_key] - self.lr * err_h)
        self.atk[away_key] = self._clamp(self.atk[away_key] + self.lr * err_a)
        self.dfn[home_key] = self._clamp(self.dfn[home_key] - self.lr * err_a)

        self.mu.setdefault(league_id, self.INIT_MU)
        self.hfa.setdefault(league_id, self.INIT_HFA)
        self.mu[league_id] += self.lr * 0.01 * (err_h + err_a)
        self.hfa[league_id] += self.lr * 0.01 * (err_h - err_a)

    def season_boundary(self, league_id: int, team_id: int, seasons_elapsed: int) -> None:
        if (team_id, self.HOME) not in self.atk or seasons_elapsed <= 0:
            return
        members = self.means.reference(league_id)
        keys = [(t, s) for t in members for s in (self.HOME, self.AWAY)]
        if keys:
            target_atk = sum(self.atk.get(k, 0.0) for k in keys) / len(keys)
            target_dfn = sum(self.dfn.get(k, 0.0) for k in keys) / len(keys)
        else:
            target_atk = target_dfn = 0.0
        keep = (1.0 - self._regression()) ** seasons_elapsed
        for slot in (self.HOME, self.AWAY):
            key = (team_id, slot)
            self.atk[key] = target_atk + (self.atk[key] - target_atk) * keep
            self.dfn[key] = target_dfn + (self.dfn[key] - target_dfn) * keep

    @staticmethod
    def _regression() -> float:
        from ratings_lib import SEASON_REGRESSION
        return SEASON_REGRESSION

    def recenter(self, league_id: int) -> None:
        # Both slots are shifted by one pooled mean rather than one each: the
        # league-wide home/away asymmetry is what `hfa` is for, and zeroing
        # the two slots separately would move that asymmetry out of hfa and
        # into the ratings, where it would be invisible to the diagnostics.
        self.means.roll_season(league_id)
        members = self.means.reference(league_id)
        if not members:
            return
        keys = [(t, s) for t in members for s in (self.HOME, self.AWAY)]
        shifts = []
        for ratings in (self.atk, self.dfn):
            shift = -sum(ratings.get(k, 0.0) for k in keys) / len(keys)
            for key in keys:
                ratings[key] = ratings.get(key, 0.0) + shift
            shifts.append(shift)
        attack_shift, defence_shift = shifts
        self.mu[league_id] = (
            self.mu.setdefault(league_id, self.INIT_MU) - attack_shift + defence_shift
        )


class PlaceboVenue(VenueAttackDefence):
    """VenueAttackDefence with the slots assigned at random per fixture.

    The two sides still take *different* slots, so the shape is identical:
    four ratings per club, each fed half that club's matches. Only the thing
    the split tracks is gone. A gain here is a fact about parameter count and
    rating variance, not about playing at home.

    Keyed on fixture_id rather than a counter so the assignment is stable
    across runs and across an archive refresh, the same reason
    league_ablate.py seeds its placebo partition from fixture ids.
    """

    def _slots(self, fixture_id: int | None) -> tuple[str, str]:
        if fixture_id is None:
            return self.HOME, self.AWAY
        return ((self.HOME, self.AWAY) if fixture_id % 2 == 0
                else (self.AWAY, self.HOME))


#: Arm name -> factory, driven together by build_venue_features.py so every
#: arm is measured on byte-identical rows.
#:
#: `hfa_served` is the control that makes the rest readable: AdaptiveHFA at
#: the served rate must reproduce ratings.csv's atkdef columns exactly, which
#: is what says the lr arms differ from v3 in the hfa rate and in nothing else.
VENUE_ARMS = {
    "hfa_served": lambda: AdaptiveHFA(hfa_rate=0.01),
    "hfa_lr10": lambda: AdaptiveHFA(hfa_rate=0.10),
    "hfa_lr100": lambda: AdaptiveHFA(hfa_rate=1.00),
    "hfa_roll": RollingHFA,
    "venue": VenueAttackDefence,
    "placebo": PlaceboVenue,
}
