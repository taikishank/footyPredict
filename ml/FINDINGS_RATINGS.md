# Findings: opponent-strength ratings

The incumbent's `team_form()` is opponent-blind. Eight matches against
relegation sides and eight against the top four produce the same 54-column
vector, because every form field is a raw per-match average with no reference
to who the matches were against. This phase asks what fixing that is worth,
and which of three standard fixes to use.

Reproduce with:

```
./inference-py/bin/python ml/build_rating_features.py --index --diagnose   # ~2 min
./inference-py/bin/python ml/rating_ablate.py --mode split                 # ~2 min
./inference-py/bin/python ml/rating_ablate.py --mode rolling               # ~17 min
./inference-py/bin/python -m pytest ml/test_ratings_lib.py
```

Nothing here touches `ml/feature_lib.py`, `ml/build_features.py`, `ml/train.py`
or `ml/train_v2.py`. The served model's training input is byte-identical; the
rating columns live in their own table joined on `fixture_id`.

---

## Headline

**Opponent strength is the largest single feature effect this project has
measured, and it needs no new ingestion at all.**

Rolling-origin validation, 19,797 fixtures over five expanding-window folds:

| arm | columns | log loss | vs. v2 | 95% CI | verdict |
|---|---|---|---|---|---|
| baseline (v2) | 56 | 0.99266 | — | — | — |
| plain Elo | 60 | 0.98675 | +0.00591 | [+0.00437, +0.00765] | clears |
| goal-difference Elo | 60 | 0.98691 | +0.00575 | [+0.00423, +0.00746] | clears |
| **attack / defence** | **63** | **0.98503** | **+0.00763** | [+0.00607, +0.00920] | **clears** |
| attack / defence, expectations only | 59 | 0.98544 | +0.00722 | [+0.00569, +0.00876] | clears |
| all three stacked | 71 | 0.98512 | +0.00754 | [+0.00593, +0.00927] | clears |

For scale, against everything else this project has tried:

| change | gain |
|---|---|
| **attack / defence ratings** | **+0.0076** |
| Phase 9's availability block (cost a lineup backfill) | +0.0061 |
| `club_goals_relative` (the v1 → v2 feature) | +0.0054 |
| the entire regularization sweep (113 points) | +0.0044 |

And on the scale that matters, since most of football's loss is irreducible.
Class priors score 1.07261 on these rows, so the whole reducible budget any
feature can compete for is **0.0876**:

| arm | share of reducible loss reached |
|---|---|
| base rates only | 0% |
| team form + `club_goals_relative` (the served v2, 56 columns) | 91.3% |
| + attack / defence ratings (7 columns) | 100% |

**Seven rating columns are worth 8.7% of everything the model knows**, against
56 columns for the other 91.3%.

---

## Attack / defence beats Elo, and that is a real difference

The five rows above are five intervals against a *common* baseline, which
cannot answer "is `atkdef` better than `elo`". Those arms share 56 columns and
score identical fixtures, so their errors are strongly correlated and their
intervals overlap whether or not the arms differ. Pairing them directly:

| arm | vs. `atkdef` | 95% CI | P(atkdef wins) |
|---|---|---|---|
| plain Elo | +0.00172 | [+0.00063, +0.00284] | 0.999 |
| goal-difference Elo | +0.00188 | [+0.00096, +0.00285] | 1.000 |
| expectations only | +0.00041 | [+0.00009, +0.00074] | 0.997 |
| all three stacked | +0.00009 | [−0.00038, +0.00057] | 0.651 |

**Attack/defence wins decisively over both Elo arms**, and stacking Elo on top
of it adds nothing — the interval spans zero. The redundancy is one-way: Elo
carries no information attack/defence lacks, while attack/defence carries
~0.0017 that Elo cannot express.

Why, mechanically: a single scalar collapses a side that wins 3-2 every week
and one that wins 1-0 every week onto the same rating. Two ratings do not.
`elo_diff` correlates +0.955 with `exp_goal_diff`, so ~9% of the latter's
variance is genuinely independent, and that 9% is where the +0.0017 lives.

### The expectations carry almost all of it

Dropping the four level columns and keeping only `exp_goals_home`,
`exp_goals_away`, `exp_goal_diff` costs +0.00041 — statistically real
(CI clear of zero) but 0.5% of the arm's value. Worth knowing because the
three expectation columns are also the cleanest on the date check below. It
buys nothing at *serving* time, though: computing an expectation requires the
per-club attack and defence ratings anyway, so this saves columns, not state.

---

## Two null results

**Margin of victory is worth nothing.** Folding the scoreline into the Elo
update — the obvious first upgrade, and the standard World Football Elo
formulation — ties plain Elo on both validation modes (+0.00575 against
+0.00591, and the two are indistinguishable when paired). A 4-0 and a 1-0
apparently say the same thing about a side once the win/draw/loss sequence is
known, at this sample size.

This is not the same as saying goals do not matter: `atkdef` is *entirely*
built on goals and wins. The distinction is that attack/defence uses the
scoreline to estimate two separate rates, while goal-difference Elo uses it
only to scale a single update — and the second is what adds nothing.

**Stacking is worth nothing.** All three arms together (71 columns) do not
beat `atkdef` alone (63). The three are near-collinear:

| | `elo_diff` | `gdelo_diff` | `exp_goal_diff` |
|---|---|---|---|
| `elo_diff` | 1.000 | 0.994 | 0.955 |
| `gdelo_diff` | 0.994 | 1.000 | 0.969 |
| `exp_goal_diff` | 0.955 | 0.969 | 1.000 |

Three formulations of opponent strength, one underlying quantity — the same
shape as Phase 9's three importance measures.

---

## Why the comparison is built the way it is

**A separate table, joined on `fixture_id`.** `build_rating_features.py` does
not rebuild `2000_window3_8.csv`. The served model's training input stays
byte-identical — the discipline Phase 9 kept — so a rating arm is compared
against the actual incumbent rather than against a baseline that moved
underneath it, and the replay can be re-run in seconds while a rating rule is
being iterated.

**Hyperparameters are imported from `train_v2`, not restated.** This measures
the rating columns and not a differently-tuned model. `train_v2.py`'s own
docstring records having introduced exactly that confound against v1.

**Rolling origin is the headline, not the 80/20 split.** Every hyperparameter
in `train_v2.py` was selected on one 7,919-row validation set — `tune.py:259`
says so itself — and testing six more arms on those same rows would compound
the problem, since the best of six arms beats a baseline by some margin even
when all six are worthless. Five expanding-window folds, none training on
anything later than it scores, give 19,797 validation fixtures and intervals
roughly half as wide.

The two modes agree on the finding and disagree on the ranking, which is
itself the argument for the rolling one: on the single split the three arms
were indistinguishable (spread 0.0009 against a CI half-width of 0.0027) and
plain Elo looked like the right pick on cost. Only the larger validation set
resolves attack/defence's lead.

**The verdict is a paired bootstrap, not a log loss ranking**, matching the
promotion bar in `PLAYER_STATS_SPEC.md` §6.

---

## What the archive forced

Two facts, from `build_rating_features.py --diagnose`:

**Five leagues, zero cross-league fixtures, zero clubs in more than one
league.** The rating pool is five fully disconnected components. Ratings are
therefore comparable only *within* a league — which is all that is needed,
since every fixture to predict is intra-league — but it means every mean,
regression target and re-centering in `ratings_lib.py` is per-league. A global
one would average over clubs that can never meet.

**116 of 211 clubs have a gap season.** A club relegated in 2015 and promoted
in 2019 would otherwise return carrying its 2015 rating. `season_boundary()`
regresses toward the league mean once per season *elapsed*, so four years away
costs four regressions and the club returns essentially at the mean — the
right prior for a side nobody in this archive has seen for four years.

**Newcomers are seeded below the league mean**, not at it. Measured rather
than assumed: clubs in their first season in the archive take 1.171 points per
game against incumbents' 1.362. Seeding them at the mean would hand a free
over-rating to every club that plays them in August.

---

## Four bugs worth recording

Three of them are the Phase 9 career-appearance failure — a column that tracks
the archive rather than the football, which scores well on a chronological
split for entirely the wrong reason — arriving by three different routes.
**None would have shown up in the log loss.** They were caught only because
`--diagnose` correlates every emitted column against the fixture date, which
exists in this phase precisely because Phase 9 recorded the original.

**1. Pool leakage — `home_elo` at r(date) = −0.42.** Seeding newcomers below
the league mean injects a deficit every August that the zero-sum update never
removes, so each league's mean bled downward across 26 seasons. Fixed by
re-centering each league at its season rollover, which preserves every rating
*difference* exactly and so is a no-op for anything the model can use.

**2. Accumulated cohort — still +0.21 after the first fix.** `LeagueMeans` was
averaging over every club ever seen in a league rather than the current twenty,
so the reference set grew for 26 years while the live league did not. Fixed by
tracking the season cohort, with the most recently finished season as the
reference at the August rollover.

**3. Home-advantage burn-in — `exp_goals_home` at +0.19.** `AttackDefence`
fitted its per-league home advantage online from 0.0, and it climbs to ~0.20
over the archive, so every early fixture got a systematically smaller home bump
than every late one. Fixed by seeding it at the level the fit settles on, which
puts it on the same footing as the Elo arms' fixed `HOME_ADVANTAGE = 65`
rather than being a fitted quantity.

All rating columns now sit under |0.15|, and the three headline difference
columns under |0.03|.

**4. `recenter()` was not rate-neutral for attack/defence.** Shifting `atk` and
`dfn` by their own means independently changes the implied goal rates, because
the two enter a rate with opposite signs and against *different* clubs — unlike
Elo's single scalar, which cancels. Caught by a test asserting `exp_goal_diff`
is unchanged across a re-centering. Fixed by moving the league baseline `mu` to
absorb the difference, which required making `mu` per-league — more correct
regardless, since scoring levels genuinely differ across these five leagues.

---

## Open items

1. **The serving shape is undecided, and is now the blocking question.**
   `team_form()` is a bounded `LIMIT 8` query; a rating is sequential state
   over the whole archive and cannot be computed by a bounded query. The two
   candidates are an in-process replay at startup (~40k fixtures, under a
   second; no migration, but mutable state per worker) and a materialized
   `team_rating` table (stateless inference and a queryable history, at the
   cost of a migration, a backfill, and an ordering problem when fixtures are
   corrected). Deferred until this phase said which arm was worth serving; it
   now has.

2. **There is no `train_v3.py` yet.** This phase measured the arms; it did not
   produce a servable artifact. `train_v2.py` is the template, and the rating
   columns would have to be built into `build_features.py`'s output (or joined
   at train time) rather than living in a side table.

3. **The hyperparameters are v2's, and may no longer be right.** `REG_LAMBDA`
   and `SUBSAMPLE` were tuned for 56 columns with no opponent-strength signal.
   Re-running `tune.py --grid reg` against the rated feature set is cheap and
   is the obvious follow-up — but it should be judged on rolling origin, not
   on the split that produced the current constants.

4. **The train/serve skew found while scoping this phase is untouched and may
   matter more than any of the above.** `inference-py/app/db.py:76`
   season-scopes team history; `ml/build_features.py` does not. Offline,
   August and September are among the *better*-predicted months (0.9917 and
   0.9843 against a 0.993 mean) because the form window still carries last
   season. Served, those same fixtures get 3-5 in-season matches. The model
   has never been evaluated on the input distribution it serves in the first
   two months of a season. Ratings are immune to this — they are explicitly
   cross-season — which may be part of why they help, but it does not fix the
   54 form columns.

5. **The draw problem is unaddressed.** The served v2 argmaxes a draw 24 times
   in 7,919 validation fixtures against a 25.2% base rate; median maximum
   probability is 0.488. Marginal calibration is good (predicted H/D/A
   `[0.430, 0.256, 0.314]` against actual `[0.436, 0.252, 0.312]`), so this is
   structural to 3-class classification rather than a calibration fault.
   `AttackDefence` winning is relevant here: it is the parameterisation a
   Dixon-Coles goals model needs, so deriving 1X2 from a score distribution is
   now a smaller step than it was.

6. **`PROMOTED_OFFSET`, `SEASON_REGRESSION` and `K` are set by convention, not
   swept.** The measured newcomer gap (1.171 vs 1.362 points per game) says the
   offset's sign and rough size are right; nothing says 60 Elo points is better
   than 40 or 80. Worth a sweep only after the serving decision, and only on
   rolling origin.
