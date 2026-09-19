"""The two rating rules an external design has that this project's does not.

FINDINGS_RWI.md §2 prices the RWI stack ingredient by ingredient and finds
five of its eight already measured here and buried - four-state venue ratings
(-0.00102), league scoring priors (-0.00001), a home-advantage prior (dead at
depth 2), a log-space goals model (shipped, as the Dixon-Coles blend), and
per-player lineup weighting (+0.0007 across three flavours in
player_stats/FINDINGS_PHASE9.md).

What is left is two mechanisms, and they share a property the rest of that
list does not: they change how a rating *absorbs a result*, not what columns
the tree sees. So both arms here are drop-in replacements for the served
`atkdef` rating, emitting AttackDefence.FIELD_ORDER's seven quantities, and
every arm carries v3's 63 columns. That is FINDINGS_VENUE.md's protocol, and
it is what makes these numbers comparable to the `venue` and `hfa_*` rows
already on the board rather than to a differently-shaped model.

  CoupledVenue     VenueAttackDefence with the two venue slots pulled toward
                   each other after every update. FINDINGS_VENUE.md tested
                   the *uncoupled extreme* and it lost, which is what a
                   variance argument predicts when each rating sees half the
                   evidence. RWI's four states are not independent - the
                   article describes a cross-venue catch-up coefficient - and
                   that middle of the axis has never been built here.

  BoundedResidual  AttackDefence with the surprise passed through a bounded
                   function before it reaches the state. `_clamp` bounds the
                   rating and therefore the runaway across many matches; it
                   does nothing about one 6-0.

Deliberately NOT an arm here: RWI's other update-rule idea, blending observed
goals with xG to form the target. build_rwi_features.py --diagnose measures
why - the archive carries no shots at all before 2015 - and FINDINGS_RWI.md §5
writes it up. A rating replayed from 2000 whose update rule changes definition
in 2015 is worse than a feature with a cutoff, because the state carries the
discontinuity forward into every fixture after it.
"""
from __future__ import annotations

import math

from ratings_lib import AttackDefence
from venue_lib import VenueAttackDefence


class CoupledVenue(VenueAttackDefence):
    """Four venue ratings per club, shrunk toward each other after each update.

    `coupling` is the share of the gap between a club's two slots removed
    every time one of them moves:

        coupling = 0    exactly VenueAttackDefence - the arm FINDINGS_VENUE.md
                        measured at -0.00102. The `couple_c0` arm below is
                        that integrity check, not a hypothesis.
        coupling = 1    both slots collapse to their mean after every match,
                        so the club carries one rating again. Note this is
                        *not* quite AttackDefence: a match writes to one slot
                        and the averaging then halves what it wrote, so the
                        c=1 end behaves like the served rating at half the
                        learning rate rather than at its own. The axis is
                        anchored at the venue end and approaches the served
                        one; it does not reproduce it, and the sweep is read
                        accordingly.

    The shrink is applied to both of a club's slots, not only the one the
    match wrote, because the point is a club-level state the two venues share
    - "this side is good" leaking into both - rather than a decay on whichever
    slot happened to play.
    """

    def __init__(self, learning_rate: float | None = None,
                 coupling: float = 0.5) -> None:
        super().__init__(learning_rate)
        assert 0.0 <= coupling <= 1.0, f"coupling out of range: {coupling}"
        self.coupling = coupling

    def _pull_together(self, team_id: int) -> None:
        if self.coupling <= 0.0:
            return
        keep = 1.0 - self.coupling
        for ratings in (self.atk, self.dfn):
            home = ratings[(team_id, self.HOME)]
            away = ratings[(team_id, self.AWAY)]
            mean = 0.5 * (home + away)
            ratings[(team_id, self.HOME)] = mean + (home - mean) * keep
            ratings[(team_id, self.AWAY)] = mean + (away - mean) * keep

    def update(self, home_id: int, away_id: int, league_id: int,
               home_goals: int, away_goals: int,
               fixture_id: int | None = None) -> None:
        super().update(home_id, away_id, league_id, home_goals, away_goals,
                       fixture_id)
        self._pull_together(home_id)
        self._pull_together(away_id)


class BoundedResidual(AttackDefence):
    """AttackDefence with `goals - lambda` passed through a bounded function.

    Two shapes, because they bound differently and the difference is the
    point:

        tanh   scale * tanh(err / scale). Smooth, and linear near zero, so
               ordinary results update exactly as they do now and only the
               tail is compressed. `scale` is the surprise at which the
               compression reaches ~24%.
        clip   max(-scale, min(scale, err)). The Huber gradient. Hard-bounded,
               and identical to the served rule inside the bound.

    At a large enough `scale` both reproduce AttackDefence to floating point,
    which is what the `bounded_lin` arm below is for: tanh's error is
    O(err^3 / scale^2), so at scale = 1e6 it is ~1e-12 on the largest surprise
    the archive contains. build_rwi_features.py asserts it against
    ratings.csv, which is what says the real arms differ from v3 in this
    function and in nothing else.

    Note the asymmetry the bound introduces, which is the arm's actual risk
    rather than an implementation detail. A Poisson residual is bounded below
    by -lambda (about -1.5) and unbounded above, so compressing the tail
    removes more from the positive side than the negative, and the ratings
    drift down. `mu` absorbs it - it is fitted online from the same residual
    and moves to keep the league's level right - but that makes `mu` and the
    bound partially confounded, and build_rwi_features.py --diagnose prints
    the fitted mu per arm so a shifted baseline is visible rather than
    silently priced into the ratings.
    """

    SHAPES = ("tanh", "clip")

    def __init__(self, learning_rate: float | None = None,
                 scale: float = 2.0, shape: str = "tanh") -> None:
        super().__init__(learning_rate)
        assert shape in self.SHAPES, f"unknown shape: {shape}"
        assert scale > 0.0, f"scale must be positive: {scale}"
        self.scale = scale
        self.shape = shape

    def _residual(self, err: float) -> float:
        if self.shape == "clip":
            return max(-self.scale, min(self.scale, err))
        return self.scale * math.tanh(err / self.scale)


#: Large enough that tanh is the identity to ~1e-12 on any residual the
#: archive can produce, and small enough that `scale * tanh(err / scale)`
#: stays well inside float precision.
LINEAR_SCALE = 1e6


#: Arm name -> factory. Driven together by build_rwi_features.py so every arm
#: is measured on byte-identical rows.
#:
#: Two integrity arms rather than one, because this phase sweeps two
#: independent rules and each needs its own anchor:
#:
#:   bounded_lin  must reproduce ratings.csv's atkdef columns exactly
#:   couple_c0    must reproduce venue_ratings.csv's `venue` columns exactly
#:
#: Both are asserted at build time. A sweep whose endpoints do not land on
#: numbers already published is a sweep measuring something other than what
#: it names.
RWI_ARMS = {
    "bounded_lin": lambda: BoundedResidual(scale=LINEAR_SCALE, shape="tanh"),
    "bounded_t05": lambda: BoundedResidual(scale=0.5, shape="tanh"),
    "bounded_t1": lambda: BoundedResidual(scale=1.0, shape="tanh"),
    "bounded_t2": lambda: BoundedResidual(scale=2.0, shape="tanh"),
    "bounded_t4": lambda: BoundedResidual(scale=4.0, shape="tanh"),
    "bounded_t8": lambda: BoundedResidual(scale=8.0, shape="tanh"),
    "bounded_c2": lambda: BoundedResidual(scale=2.0, shape="clip"),
    "couple_c0": lambda: CoupledVenue(coupling=0.0),
    "couple_c10": lambda: CoupledVenue(coupling=0.10),
    "couple_c25": lambda: CoupledVenue(coupling=0.25),
    "couple_c50": lambda: CoupledVenue(coupling=0.50),
    "couple_c75": lambda: CoupledVenue(coupling=0.75),
    "couple_c90": lambda: CoupledVenue(coupling=0.90),
    "couple_c100": lambda: CoupledVenue(coupling=1.00),
}
