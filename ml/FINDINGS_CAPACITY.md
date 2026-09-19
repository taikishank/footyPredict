# Findings: model capacity, and what it did to the rest of the board

Every hyperparameter v3 has ever been fitted with was selected on v2's 56
columns, by `ml/tune.py --grid reg` and `--grid reg2`, before `atkdef`'s seven
rating columns existed. Nothing since has re-opened them: `FINDINGS_RATINGS.md`
onward deliberately holds `train_v2`'s constants fixed so that an ablation
measures its columns and not a differently-tuned model, which is the right
default for an ablation and, it turns out, the wrong one for eight phases in a
row.

This phase re-opens them, and the answer changes two published readings, kills
both remaining small candidates, and halves the largest live finding in the
project.

The entry point was `FINDINGS_GOALS.md` open item 2. Two unrelated noise
injections had both improved v3 significantly — `FINDINGS_LEAGUE.md`'s twenty
random-partition interactions (+0.00074) and this project's three shuffled
Poisson rates (+0.00032) — and both documents read that as v3 being unable to
reach the interactions its feature set already implies. That reading predicts
v3 wants **more** capacity. It is exactly backwards.

Reproduce with:

```
./inference-py/bin/python ml/tune.py --model v3 --grid cap  --mode rolling   # ~50 min
./inference-py/bin/python ml/tune.py --model v3 --grid cap2 --mode rolling   # ~25 min
./inference-py/bin/python ml/seed_probe.py --mode rolling                    # ~20 min
./inference-py/bin/python ml/cap_runoff.py --mode rolling                    # ~12 min
./inference-py/bin/python ml/train_v3.py                                     # ~30 s
./inference-py/bin/python ml/seed_probe.py   --mode rolling   --recipe v3    # ~3 min
./inference-py/bin/python ml/goals_ablate.py --mode rolling --cadence 7 --recipe v3
./inference-py/bin/python ml/venue_ablate.py --mode rolling   --recipe v3    # ~2 min
./inference-py/bin/python ml/scope_ablate.py --mode rolling   --recipe v3    # ~1 min
./inference-py/bin/python -m pytest ml/
```

New: `ml/train_v3.py`, `ml/cap_runoff.py`, `ml/seed_probe.py`, and a `--recipe`
flag on `goals_ablate.py`, `venue_ablate.py`, `scope_ablate.py` and
`seed_probe.py`. `rating_ablate.fit_and_score` and `venue_ablate.fit_probs`
gained an optional `params`, defaulting to the recipe every published number
was measured at, so a run without the flag reproduces its FINDINGS file
unchanged. Nothing in the serving path changed.

---

## 1. Depth is monotone, in the direction nobody predicted

`--grid cap` bracketed the served `max_depth=4` on both sides and extended
toward more capacity, where the under-capacity hypothesis pointed. Under
rolling origin, marginalising over `reg_lambda`:

| max_depth | 3 | 4 (served) | 5 | 6 | 8 |
|---|---|---|---|---|---|
| log loss | **0.98390** | 0.98535 | 0.98706 | 0.98877 | 0.99217 |

Monotone increasing, across a factor of nearly three in tree size. v3 is
over-fitting at the depth it is served with. The under-capacity reading of both
placebos was dead at this point, but the winner sat on the grid's edge — the
same defect `reg` had when its optimum pinned at `reg_lambda=100` and needed
`reg2` — so `cap2` extended downward to `max_depth ∈ {1, 2, 3}`, and picked up
`n_estimators` because the two are not separable: a level of depth removed is
capacity a longer boost at the same learning rate can put back.

`cap2`'s top ten configs land within 0.00073 of each other, and its nominal
winner beats the runner-up by 0.00012. That is not a ranking anyone should act
on, which is what `ml/seed_probe.py` exists to say out loud.

## 2. The seed band, and why best-of-18 could not settle it

Eight seeds, same recipe, same rows, nothing else varied:

| recipe | SD of pooled log loss | max gap between two seeds | seed pairs "significant" |
|---|---|---|---|
| served, depth 4 | 0.00007 | 0.00021 | 3/28 (11%) |
| v3, depth 2 | 0.00006 | 0.00018 | 3/28 (11%) |

Two points. First, `cap2`'s 0.00012 lead is well inside a band that a pure
re-seed produces, so the grid's ordering at the top is not information. Second,
**11% of same-recipe seed pairs clear this project's own 95% significance bar** —
by construction every one of those is a false positive, which is the error rate
the bar actually carries at this effect size. Any published finding at or below
0.0002 should be read against that.

## 3. The placebo dies with depth — the mechanism, with the sign flipped

`ml/cap_runoff.py` re-runs `FINDINGS_GOALS.md`'s shuffled-rate placebo at each
candidate depth. If an over-capacity tree is helped by noise columns because
they absorb splits real columns would otherwise have wasted — an accidental
`colsample_bytree` — the gain must shrink to zero as the tree shrinks:

| arm | depth | placebo gain | 95% CI |
|---|---|---|---|
| `served` | 4 | **+0.00032** | [+0.00003, +0.00062] |
| `d3` | 3 | +0.00004 | [−0.00017, +0.00024] |
| `d2` | 2 | −0.00004 | [−0.00014, +0.00007] |
| `d1` | 1 | −0.00017 | [−0.00033, +0.00000] |

Monotone to zero and through it. The effect was implicit regularisation all
along, and fixing the depth removes it.

This replicates on a second, independently constructed placebo. `goals_ablate`'s
arm permutes the probability columns *within league*, preserving each league's
scoring level; `cap_runoff`'s permutes globally. At depth 4 the within-league
version scores +0.00032 [+0.00003, +0.00062]; at depth 2, −0.00004 [−0.00014,
+0.00007]. Two placebo shapes, same death.

**`FINDINGS_GOALS.md` open item 2 and `FINDINGS_LEAGUE.md` open item 1 are
closed, with the sign opposite to the one both guessed.**

## 4. The pick

Every candidate beats the served recipe, and none separates from another:

| arm | depth | `reg_lambda` | rounds | log loss | vs served | 95% CI |
|---|---|---|---|---|---|---|
| `served` | 4 | 100 | 800 | 0.98497 | — | |
| `d3` | 3 | 30 | 800 | 0.98328 | +0.00168 | [+0.00109, +0.00226] |
| **`d2`** | **2** | **30** | **800** | **0.98306** | **+0.00191** | [+0.00099, +0.00278] |
| `d2_l10` | 2 | 10 | 800 | 0.98301 | +0.00195 | [+0.00102, +0.00288] |
| `d1` | 1 | 100 | 2400 | 0.98297 | +0.00200 | [+0.00086, +0.00305] |

All six pairwise comparisons among the four leaders span zero. Log loss cannot
choose, so the tiebreakers do, and they are the ones `train_v2.py` used to take
`SUBSAMPLE=0.6` over the nominally-better 0.5: `d2` is **interior on the depth
axis** rather than sitting on the grid's edge, it needs **one third the rounds**
of `d1` to buy the same capacity, and the placebo is **already flat** there
while `d1` has pushed it negative.

`ml/train_v3.py` carries it: `max_depth=2, reg_lambda=30, n_estimators=800`,
everything else inherited from v2. It reproduces `cap2` on both protocols
(0.98614 split, 0.98313 rolling) through an independent fold loop.

## 5. Re-pricing the board

Every number in this project was measured against a model now known to be
over-fitting. The re-pricing is cheap — the feature tables and ablation scripts
all exist — and it does not leave the board where it found it.

**The blend halves.** `FINDINGS_GOALS.md`'s log pool of v3 with the Dixon-Coles
score matrix, arm `dc_h365_c7`, rolling origin:

| | depth 4 | depth 2 |
|---|---|---|
| v3 alone | 0.98497 | 0.98306 |
| blend | 0.98156 | **0.98121** |
| blend gain | +0.00341 [+0.00227, +0.00457] | +0.00184 [+0.00101, +0.00271] |
| fitted weight, mean over folds | 0.568 | 0.484 |

The two are strongly non-additive: stacking the retune's +0.00191 on the
blend's +0.00341 predicts 0.97965, and the truth is 0.98121. **Roughly half of
what the blend was credited with was the retune arriving early**, through a
pooling weight that was partly compensating for a weak first forecaster — the
weight falls exactly as predicted, in four of five folds individually.

It survives anyway. +0.00184 at P = 1.000 is not a null, it is still the largest
live finding on the board, and the blend remains the best thing measured at any
depth. The deferred question therefore resolves: **v3 is the blended
forecaster**, on log loss — though it now buys 54% of what it was promoted on,
against an unchanged serving cost, which is a trade the serving path should
re-examine rather than inherit.

**Both small candidates die.**

| candidate | depth 4 | depth 2 | verdict |
|---|---|---|---|
| `hfa_lr10` (`FINDINGS_VENUE.md`) | +0.00035 [−0.00025, +0.00097] | **−0.00017** [−0.00085, +0.00054] | null at both depths, now the wrong sign; inside the seed band |
| `xseason_age` (`FINDINGS_SCOPE.md`) | +0.00025 [+0.00001, +0.00047] | **+0.00014** [+0.00000, +0.00028] | interval touches zero; inside the seed band |

`xseason_age` needed care. `scope_ablate`'s arms are 56 columns — form plus
`club_goals_relative`, no ratings — so running them at depth 2 is not a
re-pricing but a mismatch, and indeed the whole scope board *degrades* by
~0.0022 there. The row above is instead the arm measured on **v3's actual
column set** (63 columns, plus the four staleness columns), where it is already
worth less at depth 4 (+0.00025, not the published +0.00041) because `atkdef`
had absorbed part of what staleness was buying, and worth half that again at
depth 2.

**What got clearer, not weaker.** The real score-matrix arm separates further
from its placebo once the tree stops spending splits on noise. `dc_probs`
against its own placebo goes from +0.00088 [+0.00005, +0.00171] at depth 4 to
+0.00152 [+0.00063, +0.00243] at depth 2 — it cleared zero at both depths, but
it roughly doubles, and part of that is arithmetic rather than discovery, since
the placebo it is measured against fell from +0.00032 to −0.00004. Still, both
halves move the right way: the real arm gains (+0.00121 → +0.00148 against v3)
while the noise arm dies. This is the same mechanism as §3 read from the other
side, and it is the one place in this phase where a feature looks better rather
than worse.

## 6. Serving v3, and a train/serve skew found on the way

v3's seven rating columns are not a query. `team_form` reads a club's last
eight matches, so the service can build it per request; `AttackDefence`'s state
after fixture N depends on every fixture before it, back to 2000. The plan
called for a **rating state store** — persist the ratings, have the ingestor
advance them as results land. That is the wrong design, and one measurement
settles it: **replaying the entire 26-season archive takes 0.4 seconds.**

A store would buy a fraction of a second of startup, and cost a second
implementation of a numerically delicate update rule, dual-write consistency on
every ingest, and drift that nothing could detect — a rating silently diverged
from the one the model was fitted on produces confident, wrong probabilities.
So `app/ratings_state.py` replays instead, through the identical
`ratings_lib.Replay` driver `build_rating_features.py` trains from. The offline
build was refactored onto that shared driver first, and `ratings.csv` came out
**byte-identical**, which is what makes "same loop" a checked claim rather than
an intention.

Building it surfaced a real defect. **68% of fixtures share a kickoff instant**
— a Saturday 3pm round is one timestamp — and the offline replay ordered them
with `sort_values("date")`, pandas' default **quicksort, which is not stable**.
That order is deterministic given the parquet and underivable from anything
else, so a service replaying from Postgres could not have reproduced it. It is
not a rounding difference: re-running under the two orders moves
`exp_goal_diff` by up to **1.62 on a single fixture against a column SD of
0.66**.

Both paths now sort on `(date, fixture_id)`, the one key both sources carry.
The cost of the correction, on pooled log loss:

| | before | after |
|---|---|---|
| v3 rolling | 0.98313 | 0.98314 |
| v3 split | 0.98614 | 0.98624 |

+0.00001 on rolling — inside the seed band of §2, so **no published number
needs revising**; the arbitrary tiebreak was worth nothing on average and up to
2.5 SD on individual fixtures. That asymmetry is the whole point: it would
never have shown up in a pooled metric, and it would have mispriced individual
fixtures in production indefinitely.

Serving is wired through a new `app/served.py`, so which model serves is one
config value rather than eight call sites, and `SERVED_MODEL_VERSION` still
says `v2` — nothing has been promoted. `app/ratings_state.py` rolls a *private
copy* of the state forward per request (~0.5 ms), because a season rollover
mutates whatever it touches; the replayed singleton is therefore never written
to, which is also what makes it safe to read from FastAPI's threadpool.

The equivalence is asserted, not assumed: `tests/test_ratings_state.py`
replays the archive up to three different fixtures and checks the served rating
columns against `ratings.csv` to 1e-9, and `tests/test_served.py` feeds the real
`train_v3.py` artifact a vector built by the serving path.

---

## Open items

1. **The depth-2 recipe is v3's, not the project's.** Applying it to the
   56-column scope table costs ~0.0022. Every re-pricing here was done on the
   63-column table for that reason, and `--recipe v3` should not be pointed at
   a v2-shaped arm without expecting it. The corollary is that v1 and v2 are
   now mis-tuned in the same way v3 was, and `ml/tune.py --model v2 --grid cap2`
   would say by how much — relevant because `train.py`'s incumbent is still
   what the service serves.

2. **`learning_rate` has never been swept at depth 2.** It was fixed at 0.0072
   by `--grid core` on v2's columns at depth 4, and held through `cap` and
   `cap2`. Depth and learning rate trade off the same way depth and
   `n_estimators` do, and `cap2` already showed that axis is live (`d1` needed
   2400 rounds to match `d2`'s 800). The grid is cheap now that fits cost 25
   seconds instead of two minutes.

3. **Three arms are still priced at depth 4.** The `g38_s8` window arm
   (+0.00040), the league interact arm (+0.00052) and the random-partition
   placebo (+0.00074) have not been re-run. The last of these matters most: it
   is the *other* placebo, §3 predicts it dies too, and `league_ablate.py` has
   not yet been given the `--recipe` flag the other three ablations have.
   Given how §5 went, the prior on any sub-0.0005 finding surviving is low.

4. **11% of same-recipe seed pairs clear the 95% bar.** Every paired bootstrap
   in this project resamples fixtures and holds the fit fixed, so it prices
   sampling error and not fit variance. The honest fix is to report findings
   under several seeds, which multiplies every ablation's cost by that many;
   the cheap fix is to carry the seed band alongside the interval, which
   `seed_probe.py` now makes free. Neither has been adopted as a convention.

5. **The artifact is fitted on the first 80% of the archive.** `train_v3.py`
   follows `train_v2.py` here so the shipped weights stay comparable, which
   means the model that serves has never seen the most recent fifth of its own
   history. Refitting on everything is the obvious thing to want and cannot be
   validated by construction, so it belongs to the serving path along with the
   rating state store, not to a retune.

6. **Nothing has been published or promoted.** `train_v3.py` writes
   `ml/models/match_outcome_model_v3.json` locally and uploads only under
   `--publish`, unlike `train_v2.py`, which ships on every training run. The S3
   key `2000_v3_atkdef/model.json` is still unwritten, and
   `SERVED_MODEL_VERSION` is still `v2`. Promotion is now those two steps and
   nothing else.

7. **The cached ablation artifacts predate the ordering fix.** Every
   `data/features/*_losses_*.npz` and `*_probs_*.npz` was computed against the
   old `ratings.csv`. The difference is within seed noise, so their published
   numbers stand — but a *new* arm fitted alongside a replayed cache would mix
   the two orders in one table, which is the one way this matters. The affected
   files are `rating_losses_rolling.npz`, `goals_probs_*.npz`,
   `venue_probs_*.npz`, `league_losses_*.npz` and `scope_losses_*.npz`;
   deleting them forces a clean re-fit at ~15 min each and is the safe move
   before the next ablation, not before the next read.

8. **The rating state refreshes on a timer, not an event.** `ratings_state.py`
   replays again after `RATINGS_REFRESH_SECONDS` (default an hour) because the
   service is not told when the ingestor writes. That leaves one unlucky
   request per hour paying the 0.4s replay, and a worst case of hour-old
   ratings sitting beside form columns read live. Both are acceptable at a
   fixture cadence and neither is measured.
