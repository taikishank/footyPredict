# Findings: league identity

The five leagues do not settle their matches alike, and the v3 recipe does
not know which league it is looking at. Form columns are club averages,
`club_goals_relative` divides the league's scoring level out on purpose, and
`atkdef`'s ratings are league-scoped internally but expose only per-club
numbers. That is a real hole, and the outcome rates differ across it at
p=0.001.

**Nothing here is worth shipping.** Identity is worth zero, character is worth
zero, five separate models are worth −0.0084, and the one arm that cleared its
interval was beaten by a random partition of the same shape.

Reproduce with:

```
./inference-py/bin/python ml/league_ablate.py --mode split      # ~4 min
./inference-py/bin/python ml/league_ablate.py --mode rolling    # ~25 min
./inference-py/bin/python -m pytest ml/test_league_lib.py ml/test_league_ablate.py
```

`ml/league_lib.py` is new; nothing in the serving path changed.

---

## Headline

Rolling-origin validation, 19,797 fixtures over five expanding-window folds,
every arm on top of the served v3 recipe and the same v2 hyperparameters:

| arm | cols | log loss | vs. v3 | 95% CI | verdict |
|---|---|---|---|---|---|
| **v3 (served)** | 63 | **0.98497** | — | — | — |
| `id` (league_id) | 64 | 0.98498 | −0.00001 | [−0.00016, +0.00013] | null |
| `onehot` (5 dummies) | 68 | 0.98497 | −0.00001 | [−0.00015, +0.00015] | null |
| `character` (4 rolling descriptors) | 67 | 0.98502 | −0.00005 | [−0.00040, +0.00030] | null |
| `interact` (dummies × rating expectations) | 83 | 0.98444 | +0.00052 | [+0.00016, +0.00090] | see below |
| `placebo` (**random** partition, same shape) | 83 | 0.98423 | +0.00074 | [+0.00034, +0.00113] | — |
| `per_league` (5 separate models) | 63×5 | 0.99338 | −0.00841 | [−0.01001, −0.00681] | **much worse** |

`id` and `onehot` land on −0.00001 with intervals a third the width of the
smallest effect this project has ever promoted. That is not "we could not
tell"; that is a measured zero.

---

## The intercept was dead before the ablation started

The cheapest possible version of the claim is that each league deserves its
own base rates. It is testable without fitting anything, and it fails.

Per-league outcome rates on the validation half are genuinely different —
χ²=25.6, dof=8, p=0.0012:

| league | n | home | draw | away |
|---|---|---|---|---|
| 8 | 4187 | 0.4430 | 0.2372 | 0.3198 |
| 82 | 3365 | 0.4437 | 0.2461 | 0.3103 |
| 301 | 3863 | 0.4362 | 0.2565 | 0.3073 |
| 384 | 4176 | 0.4227 | 0.2550 | 0.3223 |
| 564 | 4206 | 0.4577 | 0.2601 | 0.2822 |

But per-league class priors score **worse** than one global prior on the same
rolling folds:

```
global prior      1.07261
per-league prior  1.07308
gain             -0.00047   95% CI [-0.00097, +0.00005]
```

The reason is that a league's own rate moves further over time than the
leagues differ from each other:

| home-win rate | L8 | L82 | L301 | L384 | L564 |
|---|---|---|---|---|---|
| 2015-08..2017-10 | 0.453 | 0.472 | 0.455 | 0.469 | 0.477 |
| 2017-10..2019-12 | 0.462 | 0.444 | 0.455 | 0.432 | 0.461 |
| 2019-12..2022-02 | 0.412 | 0.431 | 0.403 | 0.405 | 0.428 |
| 2022-02..2024-04 | 0.470 | 0.452 | 0.416 | 0.423 | 0.459 |
| 2024-04..2026-09 | 0.420 | 0.420 | 0.450 | 0.387 | 0.464 |

The between-league spread is 3.5pp. League 8 alone swings 5.8pp across folds,
the ordering reshuffles every fold, and the 2019-12 row (empty stadiums) moves
all five together. **Pooling 26 years is what makes the differences
significant; it is not what makes them useful.** A significant difference and
a predictively useful one are not the same claim, and only the second one
pays.

`character` exists to answer exactly this — rolling descriptors track the
drift a dummy cannot. It is worth −0.00005. Tracking the drift correctly does
not help either, because the drift is common to all five leagues, so the
pooled model already has it in its own base rates.

## The trees did not already know

Worth ruling out before reading the nulls as "redundant". An xgboost
classifier predicting `league_id` from the 63 v3 columns reaches **28.5%**
accuracy out of time, against a 21.2% majority class and 20% chance — the
giveaways are `successful_long_passes`, `yellow_cards` and `fouls`, which is
the expected answer.

So league identity is nearly *new* information to the model, and the `id` arm
was not measuring something already present. It got zero anyway. This is the
opposite of `club_goals_relative` in `FINDINGS_WINDOW.md`, which was worth
nothing *because* `atkdef` had absorbed it.

## The one positive arm is not about football

`interact` — the five dummies times `exp_goals_home`, `exp_goals_away`,
`exp_goal_diff` — cleared its interval at +0.00052, P(better)=0.998, and beat
its own dummies by +0.00053 [+0.00024, +0.00083]. Read alone it says the
*slope* differs by league even though the intercept does not, which was the
phase's actual hypothesis and would have been a finding.

`placebo` assigns each fixture to one of five **random** groups, seeded from
nothing football knows, and builds the identical 20 product columns.

```
placebo   +0.00074  [+0.00034, +0.00113]
interact  +0.00052  [+0.00016, +0.00090]
interact over placebo:  -0.00021  [-0.00069, +0.00027]
```

**The random partition does it better.** The product columns help for a reason
that has nothing to do with which league a fixture is in — `MAX_DEPTH = 4`
(`train_v2.py`), so an interaction costs a quarter of the tree's depth, and
handing the model twenty pre-multiplied, differently-scaled copies of its
strongest signal helps whatever the multiplier is.

Without this control the phase would have shipped a league finding that was a
tree-depth artifact. It cost one extra arm and three minutes.

## Five models are much worse than one

`per_league` is the maximal version — every column interacted with league, by
construction — and it is the worst result in the table by an order of
magnitude: −0.00841, sixteen times the largest pooled effect. It loses in
**every** league, from −0.00595 (L82) to −0.00985 (L384), so this is not one
starved league dragging an average.

Each model trains on about 3,900 fixtures instead of 19,600. Whatever league
identity is worth, it is worth less than four fifths of the training data —
and the pooled table above puts a tight bound on the first quantity, so the
trade was never close.

## No league is being served badly

The last place league identity could have hidden: a single league the pooled
model systematically fails in. Raw per-league loss varies a lot —

| league | prior | v3 | Pinnacle closing | gap | share of market closed |
|---|---|---|---|---|---|
| 8 | 1.06487 | 0.97355 | 0.95213 | 0.02142 | 81.0% |
| 82 | 1.06825 | 0.99570 | 0.97696 | 0.01873 | 79.5% |
| 301 | 1.07266 | 1.00368 | 0.98376 | 0.01992 | 77.6% |
| 384 | 1.07767 | 0.97172 | 0.94681 | 0.02491 | 81.0% |
| 564 | 1.06669 | 0.97919 | 0.95925 | 0.01993 | 81.4% |

— the 0.032 spread between L384 (0.9717) and L301 (1.0037) is bigger than the
entire gap to the market. But **the market shows the same ordering**: 0.9468
against 0.9838, a 0.037 spread. That spread is how hard the leagues are, not
how badly the model handles them. Normalised, the model closes 77.6%–81.4% of
the market's information in every league, and the intervals overlap heavily.

The deficit in `FINDINGS_MARKET.md` is uniform across leagues. It is not a
league problem.

---

## Method

Arms compose by column selection on one table, and everything downstream is
`rating_ablate.py`'s machinery imported rather than restated: the same five
expanding-window folds, the same paired bootstrap on per-fixture losses, the
same `train_v2` hyperparameters. Four choices worth recording:

**`character` is causal by construction.** `league_lib.build_character()`
walks the archive in date order and calls `features()` before `update()`, the
same discipline `ratings_lib` and `LeagueRollingMean` follow.
`test_league_lib.py` builds a frame of 300 home wins and one away win and
asserts the last row's home rate is still 1.0 — a pass that updated first
would read 0.997 and would have leaked a result into its own predictors.

**A league's first 200 fixtures get NaN, not zero.** Zero would read as "a
league where the home side never wins", which is a strong and false claim;
NaN lets xgboost learn the missing branch. 2.2% of rows.

**`per_league` scatters its losses into validation order.** Every other arm
returns losses in `val_idx` order and the bootstrap pairs by position, so
concatenating per-league blocks would have silently compared different
fixtures. Tested directly.

**The placebo partition is drawn once, not per fold.** A partition redrawn
per fold would be a different feature in every model, and would have failed
for that reason rather than for the reason the control exists to test.

The single-split mode (`--mode split`) broadly agrees and is noisier, as
expected: `interact` +0.00041 [−0.00000, +0.00081], `per_league` −0.00322.
The rolling numbers are the ones to read.

---

## Open items

1. **The placebo is a lead, not just a control.** Twenty random-partition
   products of the rating expectations bought +0.00074 [+0.00034, +0.00113]
   against v3 — small, but it cleared zero, and it is free of any league
   claim. The likely explanation is that `MAX_DEPTH = 4` leaves the model
   unable to reach interactions it wants, which is a **hyperparameter**
   finding disguised as a feature one. The cheap test is a depth sweep on the
   rated feature set; `tune.py`'s grid was selected on v2 columns before
   `atkdef` existed.

2. **Away-specific team strength is still untouched** — item 1 of
   `FINDINGS_MARKET.md`, and still the best-evidenced hypothesis. Note that
   this phase is consistent with it and does not substitute for it:
   `atkdef` carries one home-advantage parameter *per league*, and the
   finding here is that per-league *anything* is worth nothing, which says
   the remaining home/away signal is per-*club*.

3. **Opponent-adjusted form statistics remain open**, carried over from
   `FINDINGS_WINDOW.md` item 3. Unaffected by anything here.

4. **The drift table is the most interesting object in this document.** All
   five leagues move together across eras, by more than they differ from each
   other, and nothing in the model is allowed to know what year it is (on
   purpose — Phase 9's career-appearance bug). Whether a *recency weighting*
   on training rows helps is untested and is the question this phase actually
   turned up.

5. **`league_losses_rolling.npz` caches losses, not probabilities** — the
   same limitation `FINDINGS_MARKET.md` item 5 and `FINDINGS_WINDOW.md` item
   4 record.
