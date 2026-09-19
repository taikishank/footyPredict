# Findings: the RWI design, priced against this board

An external model — "RWI" (Roiz Walss Index), posted to r/algobetting with a
longer write-up on LinkedIn — reports a 1X2 log loss of **0.9797** over 34,042
matches and eight seasons. Taken at face value that is ahead of everything
here, and it arrives with a design this project has spent nine phases
approaching from other directions.

This phase asks two questions in order. **Is the number comparable?** And
**which of its ingredients has this project not already measured?**

The first answer is no, and the second is: two, both inside the rating update
rule rather than in the feature set.

**Both were then measured, and both are null.** Bounding one match's surprise
is worth +0.00022 at its best scale, on an interval spanning zero and barely
outside the seed band. Coupling the venue states recovers almost everything
`FINDINGS_VENUE.md`'s four-rating arm lost and never gets past v3. A third
idea — RWI's xG-blended update target — was killed by the gate in §5 before it
was built.

The pre-registration in §4 was written before any arm was built, on the
`FINDINGS_OPPONENT.md` precedent; §5 onward is what came back.

Reproduce with:

```
./inference-py/bin/python ml/build_rwi_features.py --diagnose   # ~4 min
./inference-py/bin/python ml/rwi_ablate.py --mode split         # ~4 min
./inference-py/bin/python ml/rwi_ablate.py --mode rolling       # ~40 min
./inference-py/bin/python -m pytest ml/test_rwi_lib.py
```

New: `ml/rwi_lib.py`, `ml/build_rwi_features.py`, `ml/rwi_ablate.py`,
`ml/test_rwi_lib.py`. `ratings_lib.AttackDefence` gained one hook,
`_residual()`, which returns its argument unchanged — the served ratings are
byte-identical and `build_rwi_features.py` asserts it against `ratings.csv` on
all 40,098 fixtures. Nothing in the serving path changed.

Sources: the r/algobetting post `1s4yd7o`, and
`linkedin.com/pulse/why-most-soccer-prediction-models-ignore-1-4-outcomes-roiz-walss-qyawf`.

---

## 1. The number is not comparable, and its own table says so

RWI's reported table, as posted:

| metric | RWI | its bookmaker baseline |
|---|---|---|
| **log loss** | **0.9797** | **0.9779** |
| accuracy | 0.5245 | 0.5253 |
| macro F1 | 0.3959 | 0.3900 |
| ranked probability score | 0.1975 | 0.1974 |
| draw recall | 0.0197 | ~0 (structural) |

**It loses to its own baseline on log loss, accuracy and RPS.** The post does
not claim otherwise — it claims to be within noise of the books, and the
thread's top comment makes the obvious reply, that matching the market is a
more expensive way to buy what the market gives away.

So the raw 0.9797 cannot be read against 0.98121 here. Log loss is a property
of the fixture population, not of the model, and the two populations are
visibly different. The tell is the *baselines*:

| | forecaster | log loss | fixtures |
|---|---|---|---|
| RWI's set | its bookmaker baseline | 0.9779 | 34,042, 8 seasons, leagues unstated |
| this archive | Pinnacle closing, de-vigged | **0.96293** | 18,793 rolling-origin, five leagues |

Two "bookmaker baselines" 0.015 apart. Three explanations, and the post
distinguishes none of them: softer books than Pinnacle, no de-vigging, or a
wider and lower-tier league set than our top five. Every one of them makes
RWI's fixtures easier to sit *near the market* on, and makes the absolute
number meaningless across datasets.

The one quantity that does travel is the gap to the market each model was
measured against:

| | model | its market | gap |
|---|---|---|---|
| RWI blend | 0.9797 | 0.9779 | **−0.0018** |
| v3 blend (depth 2 + Dixon-Coles) | 0.98121 | 0.96293 | **−0.0183** |

That gap is the only genuinely interesting number in the post, and it is
unresolvable without the league list and the de-vigging treatment. If their
baseline is a vigged consensus of soft books over a wide league set, −0.0018
against it is not evidence of anything. If it is de-vigged Pinnacle on top-five
football, it is the strongest result this project has ever had to explain.

**Two further cautions on the size of the claimed edge.** The nominal gap from
0.98121 to 0.9797 is 0.0015 — and `FINDINGS_CAPACITY.md` §2 measured that 11%
of *same-recipe re-seeds* clear this project's 95% bar at that effect size.
Separately, RWI *standalone* scores 1.019, against class priors at 1.07261
here: as a forecaster it is weak, and the post is explicit that it is a feature
for a downstream blend rather than the model.

---

## 2. Ingredient by ingredient, against nine phases

RWI's stack, as described in the LinkedIn article, against what this project
has already measured. Every figure in the right column is rolling-origin,
19,797 fixtures, five expanding-window folds.

| RWI ingredient | this project's equivalent | measured |
|---|---|---|
| four states: home/away × attack/defence | `venue_lib.VenueAttackDefence`, `FINDINGS_VENUE.md` | **−0.00102** [−0.00226, +0.00024], null-to-worse |
| league-specific scoring prior | `FINDINGS_LEAGUE.md`; also `AttackDefence.mu` per league | **−0.00001**, a measured zero on a tight interval |
| home-advantage prior | `venue_lib.AdaptiveHFA`, `hfa_lr10` | +0.00035 at depth 4, **−0.00017** at depth 2 — dead |
| expected goals in log space → score matrix | `goals_lib` Dixon-Coles, log-pooled with v3 | **+0.00184** — the largest live finding here |
| blend of two forecasters | log pool, fitted weight ~0.484 | already the served recipe |
| lineup adjustment from per-player value | `FINDINGS_PHASE9.md` | absence/rotation/XI-experience **+0.0061**; per-player *importance weighting* **+0.0007** |
| ordered logit + CatBoost | XGBoost + Dixon-Coles | ordered logit untried; see §3 |
| RWI as a standalone signal | — | theirs scores 1.019, barely inside class priors |

**Five of the eight are things we have explicitly ablated and buried.** The
four-state decomposition is the one the thread singled out as thoughtful, and
`FINDINGS_VENUE.md` measured it twice — once by an ablation, once by a
reliability probe showing the club-to-club venue spread sits *below* what
sampling noise alone produces.

The lineup adjustment deserves a specific note, because it is the piece that
looks newest and is in fact the most thoroughly closed. RWI shrinks a player's
personal rating toward his positional average weighted by minutes, then applies
the lineup's aggregate as a multiplicative correction to expected goals. That
is a fourth flavour of per-player importance weighting. `FINDINGS_PHASE9.md`
tried three — minutes, goal share, and involvement — and all three landed at
about **+0.0007**, roughly a seventh of what `club_goals_relative` buys from
the scorelines alone.

---

## 3. What survives: two mechanisms, both in the update rule

Stripping out everything already measured leaves two ideas, and they have a
property in common that the rest of the list does not: they change **how a
rating absorbs a result**, not what columns the tree sees.

### 3a. Coupled venue states

`FINDINGS_VENUE.md` tested the *uncoupled extreme*. `VenueAttackDefence`
(`venue_lib.py:171`) gives a club four ratings, updated only by matches at that
venue and read only in that slot — and its own docstring names the cost: each
rating sees half the evidence, which is why `PlaceboVenue` exists.

RWI's four states are **not** independent. The article describes a *cross-venue
catch-up coefficient* pulling each state toward its counterpart. That is the
middle of an axis this project has only sampled at its endpoints:

| coupling | model | measured |
|---|---|---|
| full (one rating per club per side) | `AttackDefence`, served | 0.98306 at depth 2 |
| none (four independent ratings) | `VenueAttackDefence` | 0.98598 at depth 4, −0.00102 |
| partial | **never built** | — |

The uncoupled arm losing is exactly what a variance argument predicts, and it
says nothing about a shrunk version. This is a real gap in that ablation rather
than a new hypothesis, and it is a small change to a class that already exists.

### 3b. A bounded residual, and a shot-derived update target

`AttackDefence.update` (`ratings_lib.py:385`) clamps the **rating** and not the
**error**:

```python
err_h = home_goals - lam_h
self.atk[home_id] = self._clamp(self.atk[home_id] + self.lr * err_h)
```

`_clamp` bounds the result to `MAX_RATING = 1.5`, which stops a runaway across
many matches. It does nothing about a single one. A 6–0 against a λ of 1.5
moves an attack rating by 4.5 learning-rate units; a 2–0 moves it by 0.5. The
update is linear in the surprise, and football scorelines are not.

RWI does two things here that we do not:

1. **Bounds the residual** before it reaches the state — so one freak result
   cannot dominate.
2. **Blends the observed goals with xG** to form the update target, on the
   argument that a 1–0 from two shots and a 1–0 from twenty are different
   evidence about a team's rate.

We have no xG in the archive. We do have team-level shot counts per fixture —
`shots_total`, `shots_on_target`, `shots_off_target`, `shots_blocked`
(`feature_lib.py:56-59`) — which support a crude shot-quality proxy. That is a
weaker instrument than real xG and the arm must be read accordingly, but it is
the only ingredient in the post that attacks a defect our rule genuinely has.

### What is being left out, and why

**Ordered logit.** Untried here, and staying that way for now. An ordered model
places home/draw/away on a single latent axis, which is precisely the
assumption `FINDINGS_GOALS.md` was built to escape: the draw is not one end of
a strength axis, and a score matrix gets P(draw) off the diagonal as a
consequence of two rates rather than by carving a band out of a line. We
already have the better-structured version of what an ordered logit is for.

**CatBoost over XGBoost.** A library swap is not a hypothesis.
`FINDINGS_CAPACITY.md` §4 found four recipes within 0.0002 of each other and
could not rank them; a different gradient-boosting implementation lands in the
same band.

---

## 4. Pre-registration

Written before any arm was built and before any output was read.

### Arms

Every arm is a drop-in replacement for the rating rule, emitting
`AttackDefence.FIELD_ORDER`'s seven columns, so every arm carries v3's **63
columns** and differs from v3 in the update rule and nothing else. This is the
`FINDINGS_VENUE.md` protocol, and it is what makes the arms comparable to the
`venue` and `hfa_*` numbers already on the board.

Fitted on the v3 recipe — `max_depth=2, reg_lambda=30, n_estimators=800`
(`train_v3.py`) — not v2's, because `FINDINGS_CAPACITY.md` re-priced the whole
board at depth 2 and a depth-4 measurement is no longer the incumbent.

| arm | change | sweep |
|---|---|---|
| `v3` | served rating rule | — |
| `couple` | venue states shrunk toward each other each update | coefficient ∈ {0.1, 0.25, 0.5, 0.75} |
| `bounded` | residual through a bounded function before the state update | tanh scale ∈ {1, 2, 4}; Huber δ as a cross-check |
| `shot_target` | update target blends observed goals with a shot-derived rate | blend weight ∈ {0.25, 0.5, 0.75} |
| `shot_placebo` | same shape, shot proxy shuffled within league | — |

### Bars

1. **The placebo bar.** `shot_target` introduces data the model has never seen,
   and `FINDINGS_CAPACITY.md` §3 is the reason this is not optional: at depth 4
   a *pure noise* block scored +0.00032 with an interval clear of zero, and the
   effect was implicit regularisation that died at depth 2. `shot_target` must
   beat `shot_placebo`, not merely beat `v3`.

2. **The seed bar.** `FINDINGS_CAPACITY.md` §2 measured the same-recipe seed
   band at SD 0.00006 with 11% of seed pairs clearing the 95% interval. **No
   arm under +0.0002 will be promoted**, whatever its interval says. `couple`
   and `bounded` are both expected to be small, and this bar is written down
   first precisely because they are.

3. **The interior bar.** `FINDINGS_CAPACITY.md` §4's tiebreaker: a winner
   pinned at the edge of its sweep is a sweep that was too narrow, not a
   result. Any optimum on a boundary extends the grid before it is read.

### Protocol

Rolling origin, five expanding-window folds, the same 19,797 fixtures every
number in `FINDINGS_VENUE.md` and `FINDINGS_LEAGUE.md` is measured on. Paired
bootstrap against v3 on identical fixtures. Probabilities cached, not losses —
`FINDINGS_VENUE.md` open item 4.

### Stop conditions

- `couple` monotone in the coefficient toward the served end (i.e. best at the
  most-coupled point, converging on `AttackDefence`) closes the venue axis for
  good rather than inviting a finer sweep.
- `shot_target` failing to separate from `shot_placebo` ends the xG-proxy line
  here; a real xG feed would be a new ingestion and a different phase.
- All three arms inside the seed band stops the phase, and the answer to the
  post is §1 and §2 alone.

### Everything above is the plan as written. What actually changed

Recorded here rather than by editing the plan, so the two stay legible apart.

- **`shot_target` and `shot_placebo` were never built.** The gate refused them
  (§5). That is the stop condition firing, not a scope cut.
- **Both surviving sweeps were extended once, under bar 3.** Both optima came
  back on a grid edge on the first pass, so `bounded` gained scales 0.5 and 8
  and `couple` gained coefficients 0.90 and 1.00. Neither extension changed a
  verdict; both moved the optimum to the interior, which is the only condition
  under which bar 3 allows the curve to be read at all.
- **`bounded_lin` turned out to be the sweep's right-hand endpoint, not just
  an integrity arm.** tanh at `LINEAR_SCALE` *is* the scale → ∞ point, so the
  bounded axis was closed on the right by a number already measured. Leaving it
  out of the reported sweep made an interior optimum look like an edge one.
- **`couple_c0` is reported, not asserted,** against `venue_ratings.csv`. §7.

---

## 5. The gate fired: there are no shots before 2015

`build_rwi_features.py --diagnose` asks whether the archive can support a
shot-derived update target at all. Share of each season's fixtures carrying
both `shots_total` and `shots_on_target` for **both** sides:

| season | coverage | | season | coverage |
|---|---|---|---|---|
| 2000–2005 | 0.0% | | 2015 | 62.6% |
| 2006–2007 | ~1% | | 2016 | 97.8% |
| 2008 | 7.2% | | 2017–2019 | 100% |
| 2009–2014 | 1–5% | | 2021–2025 | 100% |

Zero before 2006, negligible through 2014, and a step change in 2015–16.
**The arm was not built**, which is the pre-registered stop condition and not
a judgement call made after seeing a number.

The reason it is a hard gate rather than a cutoff to live with is worth
stating, because it is the one methodological point in this phase that
generalises. A *feature* with a coverage cutoff can be measured on the seasons
that have it — that is exactly what `FINDINGS_PHASE9.md` did, at a cost of half
the training history. A *rating* cannot. The state replays from 2000 and every
fixture inherits it, so a rule that changes definition in 2015 carries the
discontinuity forward into every row afterwards, and there is no validation
slice on which the rule was consistent. The 2016+ subset does not rescue it;
it hides it.

This also re-reads the post. RWI's eight seasons all sit inside the era where
shot and xG data is universal, so the design has never had to survive a
26-season archive. That is not a criticism of it — it is a reason its
ingredient list transfers to this project worse than it looks like it should.

---

## 6. Both surviving arms are null

Rolling origin, 19,797 validation fixtures over five expanding-window folds,
every arm carrying v3's 63 columns and fitted at `train_v3.PARAMS`
(`max_depth=2, reg_lambda=30, n_estimators=800`).

### Bounding the surprise

`bounded_lin` is tanh at a scale where it is the identity, so it is both the
integrity check (it reproduces v3 to a max per-fixture loss delta of exactly
0) and the scale → ∞ end of the axis:

| arm | tanh scale | log loss | vs. v3 | 95% CI | P(better) |
|---|---|---|---|---|---|
| `bounded_t05` | 0.5 | 0.98505 | −0.00196 | [−0.00331, −0.00065] | 0.002 |
| `bounded_t1` | 1 | 0.98351 | −0.00042 | [−0.00140, +0.00054] | 0.184 |
| `bounded_t2` | 2 | 0.98307 | +0.00002 | [−0.00056, +0.00060] | 0.504 |
| **`bounded_t4`** | **4** | **0.98287** | **+0.00022** | [−0.00008, +0.00054] | 0.926 |
| `bounded_t8` | 8 | 0.98299 | +0.00010 | [−0.00006, +0.00025] | 0.876 |
| `bounded_lin` | ∞ | 0.98309 | +0.00000 | — | — |

A shallow interior maximum at scale 4, decaying smoothly back to zero as the
bound is removed — a well-behaved curve, and one whose peak is +0.00022 with
an interval through zero. Against the 0.0002 seed band (`FINDINGS_CAPACITY.md`
§2, where 11% of same-recipe seed pairs clear the 95% bar), that is not a
finding. **Bar 2 refuses it.**

The `clip` cross-check agrees, which matters: the Huber gradient at the same
scale lands within −0.00001 [−0.00033, +0.00035] of tanh. Two bound shapes at
one scale being indistinguishable, while the scale sweep moves 0.002 from end
to end, says the axis that does anything is *how much* signal is removed, not
the bounding.

And the direction at the aggressive end is the arm's own refutation. `t05`
loses 0.002, and the by-outcome split says where: away-win loss rises from
1.04978 to 1.06759 while home-win loss *falls*. The predicted marginals move
the same way — P(home win) 0.4536 → 0.4587 against an actual 0.4406. Bounding
the residual does not make the rating robust; it starves it, and because a
Poisson residual is bounded below by −λ and unbounded above, what it starves
first is evidence of a strong away side. That is `FINDINGS_VENUE.md` open item
2 getting worse, not better.

**The premise was right and the fix is not worth having.** The update genuinely
is linear in a surprise that football does not distribute symmetrically. It
turns out the served learning rate (0.02) is already small enough that one 6–0
moves a rating by ~0.09 on a log scale, and `_clamp` catches the runaway case.
There was less to fix than the mechanism suggested.

### Coupling the venue states

| arm | coupling | log loss | vs. v3 | 95% CI | P(better) |
|---|---|---|---|---|---|
| `couple_c0` | 0.00 | 0.98642 | −0.00333 | [−0.00479, −0.00190] | 0.000 |
| `couple_c10` | 0.10 | 0.98332 | −0.00023 | [−0.00123, +0.00077] | 0.317 |
| `couple_c25` | 0.25 | 0.98320 | −0.00011 | [−0.00100, +0.00076] | 0.401 |
| **`couple_c50`** | **0.50** | **0.98317** | **−0.00008** | [−0.00095, +0.00075] | 0.408 |
| `couple_c75` | 0.75 | 0.98326 | −0.00017 | [−0.00102, +0.00065] | 0.338 |
| `couple_c90` | 0.90 | 0.98331 | −0.00022 | [−0.00106, +0.00060] | 0.291 |
| `couple_c100` | 1.00 | 0.98344 | −0.00035 | [−0.00120, +0.00043] | 0.191 |

**No arm beats v3 at any coupling.** The curve is read as a curve, per bar 1,
and what it shows is unambiguous: the uncoupled end loses 0.0033, ten percent
of coupling recovers 93% of that, and the remainder of the axis is flat inside
the seed band. A shallow interior optimum at 0.50 that is still *negative*
against v3.

That shape is the answer to the hypothesis. If the venue split carried a real
club trait, partial coupling would trade a little of it for a lot of variance
and land somewhere *above* v3 — a trait plus better-estimated ratings beats
ratings alone. Instead the curve climbs to v3's level and stops there, which
is what a pure variance-reduction curve does when there is nothing underneath
it. `FINDINGS_VENUE.md` measured the venue split dead; this measures the whole
axis between it and the served rating dead, at every point on it.

Note `couple_c100` is slightly *worse* than `c50`, as `rwi_lib.CoupledVenue`'s
docstring predicted before the run: at full coupling a match writes one slot
and the averaging immediately halves it, so that end behaves like the served
rating at half the learning rate rather than like the served rating.

**RWI's cross-venue catch-up coefficient is not the reason its four states
work.** Whatever that design is getting from them, this archive does not have
it at any coupling.

---

## 7. A defect found on the way: two replay orders

The `bounded_lin` assert failed on its first run, by 1.6245 — and that number
is already written down in this repo. `build_rating_features.order()` documents
it exactly: 68% of fixtures share a kickoff timestamp, so sorting the index on
`date` alone leaves ~27k rows in pandas' quicksort order, and re-running the
replay under the two orders moves `exp_goal_diff` by up to 1.62 against a
column SD of 0.66.

`build_rating_features.py` sorts on `(date, fixture_id)` with a stable
mergesort, which is reproducible and which `app/db.py`'s
`get_results_index()` can match with an `ORDER BY`. **`build_venue_features.py`
and `build_goals_features.py` both sort on `date` alone.**

So `ratings.csv` is on the reproducible order and `venue_ratings.csv` and the
goals table are not. This replay uses `order()`, because a rating rule that
might be served has to; that is why `couple_c0` is reported against
`venue_ratings.csv` at a delta of 1.654 rather than asserted to zero, while
`bounded_lin` is asserted against `ratings.csv` at 1.3e-11.

Nothing here invalidates `FINDINGS_VENUE.md` or `FINDINGS_GOALS.md` — the
order is arbitrary, not wrong, and `order()`'s own note puts the mean absolute
move at ~1% of an SD. But those two tables cannot be rebuilt reproducibly, and
a phase that re-uses them alongside `ratings.csv` is mixing two replays. See
open item 3.

---

## Open items

1. **The comparison cannot be settled from the post.** What would settle it is
   the league list and whether the baseline is de-vigged. Both are one question
   to the author. Until then −0.0018 is uninterpretable and 0.9797 is not a
   target.

2. **The draw is the one place the post's instinct matches ours.**
   `FINDINGS_MARKET.md` puts 35.5% of the gap to the closing line on draws, and
   RWI's stated motivation is the same observation. But 1.97% draw recall
   against a 25.4% draw rate is not a solution to it — it is the same
   structural failure one order of magnitude less severe. Our Dixon-Coles
   diagonal is a better-founded attack on the identical problem, and it is
   already shipped.

3. **Two replay orders exist and one of them is not reproducible.** §7.
   `build_venue_features.py` and `build_goals_features.py` should take
   `build_rating_features.order()`. It is a two-line change each, it would let
   `couple_c0` be an assert rather than a report, and it is a precondition for
   any phase that joins those tables to `ratings.csv` and treats the result as
   one replay. The published numbers in those two documents would move by ~1%
   of an SD on the affected rows; nothing in the serving path depends on
   either table.

4. **The away-win share of the gap is still unexplained**
   (`FINDINGS_VENUE.md` open item 2), and this phase makes it worse rather than
   better. `bounded_t05` degrades away-win loss specifically (1.04978 →
   1.06759) while improving home-win loss, which is the Poisson residual's
   asymmetry showing through. Whatever identifies a strong away side to the
   closing line is now measured as *not* a venue parameter, *not* a
   calibration constant, and *not* a robustness treatment of the update.

5. **The venue axis is closed.** Three phases have now attacked it —
   `FINDINGS_VENUE.md`'s reliability probe and ablation, and this one's sweep
   of every point between the split and the served rating. It should not be
   re-opened without a new data source, and a design that appears to succeed
   with venue-split ratings elsewhere is not one.

6. **What is still worth chasing is unchanged by this phase.**
   `FINDINGS_MARKET.md`'s 0.0183 to the closing line, with 35.5% of it on
   draws, and `FINDINGS_CAPACITY.md`'s note that the Dixon-Coles blend now
   buys 54% of what it was promoted on against an unchanged serving cost. The
   external design contributed no candidate that survived contact with this
   archive.
