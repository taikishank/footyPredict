"""The two form-window scoping recipes the project currently runs at once.

`ml/build_features.py` keeps one unbounded list of prior matches per club and
averages the last MAX_WINDOW of them, whatever season they fall in.
`inference-py/app/db.py:76` scopes the same query to the fixture's own
`season_id`. Both have been true since the v2 model shipped, so the served
model is fitted on one input distribution and queried on another.

Replaying the archive says how far apart they are:

    20.4%  of training rows carry at least one prior-season match in their
           16 window slots
     7.4%  of training rows (2,936) cannot be served at all - neither club
           has MIN_WINDOW matches inside the current season
    11.6%  of August fixtures are servable; September 82.4%, October 99.2%
     2.0%  of training rows average a *partial* window (n < MAX_WINDOW)
           against ~100% of served early-season rows, because team_form()
           divides by len(window)

So the skew is two defects. Coverage: 88% of August has no served prediction
while the offline table scores it happily. Distribution: a 3-match mean is a
legitimate but much noisier input that the model saw on 2% of its training
rows and emits on every early-season fixture. The good offline August number
(0.9917 against a 0.993 mean) is measured on eight matches of carried-over
form the service does not have.

This module is the recipes only - which prior matches enter the window, and
how many a club needs before it gets a row. Everything downstream
(team_form, club_goals_relative, the league reference) is unchanged and
shared, so an arm difference is a scoping difference and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

from feature_lib import MAX_WINDOW, MIN_WINDOW

# Per-side columns that tell the model how stale its own window is. Only the
# `xseason_age` arm carries them: if the skew is really "the model cannot
# tell August form from March form", then naming the staleness should recover
# most of it without changing which matches the window holds - a strictly
# cheaper fix than either realignment, since it touches no query.
STALENESS_FIELDS = ("matches_into_season", "stale_slots")


@dataclass(frozen=True)
class Recipe:
    """cross_season: keep prior-season matches in the window (the training
    recipe) or drop them (the serving recipe).

    min_matches: how many in-window matches a club needs before the fixture
    gets a row at all. MIN_WINDOW is what both paths use today; MAX_WINDOW is
    the `scoped_full8` arm, which refuses to predict rather than average a
    partial window.

    form_source: which of the two emitted form column sets the arm reads.
    Two recipes differing only in `min_matches` share one set of columns and
    differ only in which rows they are eligible for, so the build emits two
    column sets, not four.
    """

    cross_season: bool
    min_matches: int
    form_source: str
    staleness: bool = False


RECIPES: dict[str, Recipe] = {
    # The training recipe, unchanged - the arm every published number in this
    # project was measured under.
    "xseason": Recipe(cross_season=True, min_matches=MIN_WINDOW, form_source="xseason"),
    # The same window, plus two columns naming how much of it is last season.
    "xseason_age": Recipe(
        cross_season=True, min_matches=MIN_WINDOW, form_source="xseason", staleness=True
    ),
    # The serving recipe, exactly as db.py runs it: in-season only, partial
    # windows averaged over whatever exists down to MIN_WINDOW.
    "scoped": Recipe(cross_season=False, min_matches=MIN_WINDOW, form_source="scoped"),
    # In-season only and no partial windows ever. Separates "carried-over form
    # was load-bearing" from "the partial window is the problem", by paying
    # for it in coverage: no club has eight in-season matches before October.
    "scoped_full8": Recipe(cross_season=False, min_matches=MAX_WINDOW, form_source="scoped"),
}

BASELINE_ARM = "xseason"


def window(history: list[dict], season_id: int, *, cross_season: bool,
           size: int = MAX_WINDOW) -> list[dict]:
    """The prior matches a recipe's form vector averages, most-recent-last.

    `history` is most-recent-last and each entry carries the `season_id` of
    the match it records. The cross-season branch is `history[-size:]`, which
    is build_features.py's line; the scoped branch filters first, which is
    db.py's `(raw ->> 'season_id')::bigint = %s` before its `LIMIT`.
    """
    if not cross_season:
        history = [h for h in history if h["season_id"] == season_id]
    return history[-size:]


def staleness(history: list[dict], season_id: int, size: int = MAX_WINDOW) -> dict[str, int]:
    """How much of a cross-season window is last season's football.

    `matches_into_season` counts every in-season match played, not just the
    ones inside the window, so it keeps rising after the window is full - it
    is "how far into the season are we" as the club experienced it, which is
    what a model would need to discount an August vector. `stale_slots`
    counts prior-season entries inside the window itself, and is 0 from the
    ninth in-season match on.
    """
    in_season = sum(1 for h in history if h["season_id"] == season_id)
    stale = sum(1 for h in history[-size:] if h["season_id"] != season_id)
    return {"matches_into_season": in_season, "stale_slots": stale}


def eligible(history: list[dict], season_id: int, recipe: Recipe) -> bool:
    """Whether one club clears `recipe`'s gate for a fixture in `season_id`."""
    return len(window(history, season_id, cross_season=recipe.cross_season)) >= recipe.min_matches


def fixture_eligibility(home_history: list[dict], away_history: list[dict],
                        season_id: int) -> dict[str, bool]:
    """Per-arm "would this arm emit a row for this fixture", both clubs gated.

    Recorded per fixture at build time rather than re-derived later, because
    the arms deliberately cover different row sets and the ablation needs to
    report both the intersection (where a paired bootstrap is valid) and each
    arm's own coverage (where refusing to predict has to be priced).
    """
    return {
        name: eligible(home_history, season_id, r) and eligible(away_history, season_id, r)
        for name, r in RECIPES.items()
    }
