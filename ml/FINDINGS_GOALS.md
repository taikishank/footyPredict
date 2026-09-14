# Findings: the Dixon-Coles goals model

Every model this project has fitted classifies the outcome. This phase asks
what happens if one predicts the *scoreline* instead and reads 1X2 off the
score matrix, which is the only remaining item that changes the shape of the
model rather than its inputs — `FINDINGS_OPPONENT.md` open item 3, after four
consecutive feature nulls.

The motivating number is `FINDINGS_MARKET.md`'s outcome split: **35.5% of the
gap to Pinnacle's closing line is on draws**, and a classifier is badly placed
to fix that. The draw is not one end of a strength axis, so no monotone
function of a rating gap picks it out; a score matrix gets P(draw) from the
diagonal instead, as a structural consequence of two goal rates.

Reproduce with:

```
./inference-py/bin/python ml/build_goals_features.py             # ~4 min
./inference-py/bin/python ml/build_goals_features.py --diagnose  # seconds, cached
./inference-py/bin/python ml/goals_ablate.py --mode split        # ~5 min
./inference-py/bin/python ml/goals_ablate.py --mode rolling      # ~35 min
./inference-py/bin/python -m pytest ml/test_goals_lib.py
```

New: `ml/goals_lib.py`, `ml/build_goals_features.py`,
`ml/goals_ablate.py`, `ml/test_goals_lib.py`. Nothing in the serving path
changed.

---

## The probe, before anything was built

`ratings_lib.AttackDefence` already emits two Poisson goal rates per fixture,
and they are already in `ratings.csv`. So the cheapest possible version of
this phase — a score matrix over rates the project has had all along — cost
seconds and needed no new machinery. `FINDINGS_OPPONENT.md` open item 5 is why
it ran first.

19,797 rolling-origin validation fixtures, the same rows every number in
`FINDINGS_VENUE.md` is measured on:

| forecaster | inputs | log loss | draw separation |
|---|---|---|---|
| independent Poisson (ρ = 0) | 2 rates | 0.98758 | +0.0095 |
| Dixon-Coles (ρ = −0.08) | 2 rates | 0.98673 | +0.0102 |
| **v3 (served)** | **63 columns** | **0.98497** | **+0.0119** |
| Pinnacle closing | — | 0.96293 | +0.0176 |

Two things fell out of it, and they point in opposite directions.

**The two rates are almost the whole model.** A score matrix over two numbers
lands 0.0018 behind the entire 63-column feature table. Whatever v3 knows, it
is very nearly all in `atkdef` — which makes a *better pair of rates* the
highest-leverage object this project has left, and is the strongest argument
the phase has for continuing.

**The τ correction is real and small.** Fitting ρ on the pre-validation rows
gives −0.08 — the expected sign, since low-scoring draws are more common than
independence predicts — and it is worth +0.00085 standalone, 4% of the market
gap.

**But the motivating hypothesis is already in trouble.** The whole reason to
build a goals model was the draw, and at this fit quality the score matrix
picks draws *worse* than the classifier it was meant to beat: +0.0102 against
v3's +0.0119, with the market at +0.0176. The geometry alone does not deliver
it.

What the probe does **not** test is the Dixon-Coles *fit*. Those rates come
from an online gradient step per fixture, tuned as a feature generator. The
paper solves a time-weighted likelihood over the recent past instead, and that
is the untested half.

---

## Pre-registration

Written before the batch replay was run and before any of its output was read.
Reproduced here from `build_goals_features.py`'s docstring, which is where it
was committed first.

The phase escalates to the rolling ablation if **either** holds:

1. **Better rates exist.** The best (half-life, cadence) arm's standalone log
   loss beats the online-rate DC baseline of 0.98673 by at least **0.0010** —
   ~5% of the 0.0211 market gap, and about a quarter of the smallest effect
   this project has promoted.

2. **The draw mechanism works after all.** The best arm's draw separation
   reaches v3's **+0.0119**.

If neither holds, the phase stops at the diagnose and the ablation is not run,
on `FINDINGS_OPPONENT.md`'s precedent.

Arms: half-life ∈ {180, 365, 730} days × refit cadence ∈ {7, 30, 90} days.
The half-life is Dixon and Coles' own time-weighting parameter, bracketed
rather than adopted — their archive was four English divisions in the early
nineties. The cadence is swept rather than chosen because 7 days is the
literal paper (refit every matchday) and the question is whether that fidelity
is worth 13× the refits over 26 seasons.

Independently of the gate, every emitted column is checked for correlation
with the fixture date; |r| ≥ 0.10 is an era proxy, the failure mode that cost
the ratings phase three bugs.

---

## The gate cleared on the fit, not on the draw

```
1. better rates: best arm dc_h365_c7 at 0.98350 vs the online rates' 0.98673
   gain +0.00323, bar +0.00100  ->  PASS
2. draw mechanism: sharpest arm dc_h365_c30 separates +0.0113 vs v3's +0.0119
   bar: reach v3  ->  fail
```

Standalone log loss on the 19,797 rolling-origin validation fixtures — no
tree, no form column, two fitted parameters per club:

| half-life | cadence 7d | cadence 30d | cadence 90d |
|---|---|---|---|
| 180 days | 0.98591 | 0.98670 | 0.99022 |
| **365 days** | **0.98350** | 0.98417 | 0.98670 |
| 730 days | 0.98503 | 0.98564 | 0.98753 |

Both sweeps have an interior optimum, which is the shape that says a parameter
was worth sweeping. A year of half-life beats half a season and two seasons.
Refit cadence is monotone — 7 beats 30 beats 90 at every half-life — and costs
about what the half-life choice does: 0.0032 from 7 to 90 days at the best
half-life, against 0.0024 across half-lives at the best cadence. Both are
live parameters, and **at a 90-day cadence the whole exercise evaporates**:
the best arm there scores 0.98670, which is the 0.98673 the online rating
already delivered. A batch fit that is three months stale is not worth
solving for.

**Solving the likelihood beats stepping it.** +0.00323 over the same score
matrix driven by `AttackDefence`'s online rates — 15% of the market gap, and
the largest movement anything in this project has produced since the ratings
phase, though it is a score-model-against-score-model comparison rather than
an ablation gain against v3. What that is worth to the served model is the
next section's question.

**The draw hypothesis is dead in its second and last form.** The batch fit
moved draw separation from the online rates' +0.0102 to +0.0113 and still did
not reach v3's +0.0119, with the market at +0.0176. The score matrix does not
pick draws better than the classifier does. Everything the phase found is in
the rates.

---

## Two columns match sixty-three

The line worth pausing on: `dc_h365_c7` scores **0.98350 standalone** against
v3's **0.98497**. A model with two parameters per club, no form table, no
25 per-match statistics and no tree lands level with the entire served recipe.

The paired bootstrap is what keeps this honest:

```
dc standalone vs v3:  +0.00147  [-0.00046, +0.00354]  P(dc better) = 0.929
```

That interval spans zero. The claim is **parity, not superiority** — but
parity is the striking half. Everything the form table contributes over two
Poisson rates is inside the noise of this archive.

**It is not a refit-recency artifact.** The obvious objection is that DC
refits every 7 days while v3 is fitted once per fold and is up to 2.6 years
stale by the end of one. If that were the explanation, DC's edge would grow
across each fold. It does the opposite:

| position in fold | n | v3 | dc | dc gain |
|---|---|---|---|---|
| first quarter | 4950 | 0.98480 | 0.98276 | +0.00204 |
| second | 4950 | 0.99377 | 0.99233 | +0.00144 |
| third | 4950 | 0.97478 | 0.97291 | +0.00186 |
| last quarter | 4947 | 0.98652 | 0.98599 | +0.00053 |

---

## The date-correlation check needed fixing, not the columns

`--diagnose` flagged `lam_away` (+0.16) and `p_draw` (−0.20) on all nine arms,
against the project's |r| < 0.10 bar — the check that caught three bugs in the
ratings phase and Phase 9's career-appearance bug before it.

**The columns are right and the threshold was wrong.** A fitted rate is a
smoothed conditional mean with a fraction of the variance of the outcome it
estimates, so the *same* trend explains 0.4% of the raw variance and 3% of the
estimate's. Away scoring really did rise across this archive: actual away
goals go 1.083 → 1.279 across eras (raw r = +0.061), and `lam_away` goes
1.098 → 1.243 (r = +0.164). Same football, same magnitude, very different r.

`build_goals_features.py` now runs `era_drift` alongside the correlation, which
compares each column's drift against the drift in the quantity it estimates:

| column | model drift | actual drift | ratio |
|---|---|---|---|
| `lam_home` | +0.026 | +0.017 | 1.51 |
| `lam_away` | +0.146 | +0.134 | 1.09 |
| `p_home` | −0.023 | −0.031 | 0.74 |
| `p_draw` | −0.010 | −0.015 | 0.68 |
| `p_away` | +0.034 | +0.047 | 0.72 |

Every ratio has the right sign, and the probability columns *under*-shoot the
real drift. Phase 9's appearance counter would score near zero here: it climbed
with the archive while the thing it claimed to measure did not move at all.

## An independent confirmation of the venue phase

The fitted league-level parameters are diagnostics — nothing reads them — but
one of them is worth recording. `dc_hfa`, fitted by maximum likelihood with no
knowledge of any earlier phase:

| era | ρ | log home advantage |
|---|---|---|
| 2000–04 | −0.052 | 0.309 |
| 2010–14 | −0.078 | 0.322 |
| 2015–19 | −0.051 | 0.285 |
| 2020–24 | −0.066 | 0.221 |
| 2025– | −0.050 | 0.201 |

`FINDINGS_VENUE.md` measured home advantage falling "0.30 → 0.20 on the log
scale while `AttackDefence` carried a near-frozen 0.24", by a completely
different route. This estimator reproduces it to two decimal places. Two
independent estimators agreeing is worth more than either alone, and it
upgrades that phase's open item 3 from a plausible half-tick to a
twice-measured bias.

ρ settles around **−0.06**, stable across 26 seasons, against Dixon and Coles'
−0.13 on early-nineties English football. The correction is real, it has the
sign they reported, and it is about half the size in this archive.

---

## What it is worth to the served model

Rolling-origin, 19,797 fixtures, five expanding-window folds, `train_v2`
hyperparameters throughout, arm `dc_h365_c7`:

| arm | cols | log loss | vs. v3 | 95% CI | verdict |
|---|---|---|---|---|---|
| **v3 (served)** | 63 | **0.98497** | — | — | — |
| `dc_rates` (+ the two fitted rates) | 65 | 0.98399 | +0.00098 | [+0.00049, +0.00147] | real |
| `dc_probs` (+ the three matrix probabilities) | 66 | 0.98376 | +0.00121 | [+0.00044, +0.00200] | real |
| `dc_full` (both) | 68 | 0.98359 | +0.00137 | [+0.00057, +0.00219] | real |
| `dc_swap` (score model replacing `atkdef`) | 61 | 0.98425 | +0.00071 | [−0.00038, +0.00187] | null |
| `placebo` (**shuffled** rates, same shape) | 66 | 0.98464 | +0.00032 | [+0.00003, +0.00062] | **see below** |
| **`blend` (log-linear pool, outside the tree)** | — | **0.98156** | **+0.00341** | **[+0.00227, +0.00457]** | **real** |

### The tree is the wrong place to put it

Every column arm pays something and all of them are small. The blend — v3's
own probabilities pooled with the score model's in log space, one weight
fitted per fold on a held-out tail of that fold's training slice — is worth
**2.5× the best column arm**. Same two forecasters; the only difference is
whether a gradient-boosted tree has to learn the combination from 19,797 rows
or whether one parameter does it explicitly.

Fitted weights per fold: 0.58, 0.75, 0.60, 0.53, 0.38. The score model takes
more than half the weight in four folds of five, which is the ablation's
version of the standalone parity result.

**+0.00341 is 16% of the 0.0211 gap to the closing line**, and the largest
gain this project has measured since the ratings phase's +0.0076.

It survives the arm choice. `dc_h365_c7` was picked as the best of nine on the
same validation rows the ablation scores, so the same weights were replayed
against every other arm:

| arm | blended | vs. v3 |
|---|---|---|
| `dc_h365_c7` | 0.98156 | +0.00341 |
| `dc_h730_c7` | 0.98172 | +0.00324 |
| `dc_h365_c30` | 0.98179 | +0.00318 |
| `dc_h730_c30` | 0.98193 | +0.00303 |
| `dc_h180_c7` | 0.98293 | +0.00204 |
| `dc_h180_c90` | 0.98489 | +0.00008 |

Seven of nine arms clear zero and the four sensible ones sit within 0.0004 of
each other. The result is a property of blending a batch-fitted score model,
not of the particular arm selected.

### The score matrix earned nothing; the rates earned it all

```
dc_probs against dc_rates:  +0.00023  [-0.00031, +0.00084]  P = 0.779
```

Handing the tree the three probabilities rather than the two rates it was
already given is a measured null. Everything the column arms bought is the
**fit** — a better pair of Poisson rates — and nothing is the score matrix's
geometry. Combined with the failed draw gate, that is the phase's verdict on
Dixon-Coles as a *shape*: it is worth +0.00085 standalone and it is worth zero
to the served model.

### The placebo cleared zero, for the second time

```
placebo (shuffled rates): +0.00032 [+0.00003, +0.00062]
dc_probs against its placebo: +0.00088 [+0.00005, +0.00171]  P = 0.985
```

Three columns of *shuffled* rates improve v3 by a small but significant
amount. `FINDINGS_LEAGUE.md` open item 1 recorded the same thing — twenty
random-partition interactions worth +0.00074 — and guessed it was a
hyperparameter finding disguised as a feature one. This is the second
independent instance, now with a placebo of a completely different shape, and
it makes that guess much harder to dismiss: **v3 is regularised such that
adding noise columns helps it.**

It also means the column arms must be read against the placebo rather than
against v3. `dc_probs` beats its own placebo by +0.00088, so it is real — but
a quarter of its headline number was shape rather than information. The blend
is immune to this entirely: no tree, no extra columns.

### Where the gain lands

Per-fixture loss by actual outcome, `FINDINGS_MARKET.md`'s own split:

| arm | home_win | draw | away_win | pooled |
|---|---|---|---|---|
| v3 | 0.71024 | 1.38950 | 1.04813 | 0.98497 |
| `dc_probs` | 0.71389 | 1.38887 | 1.03951 | 0.98376 |
| **`blend`** | 0.72023 | **1.37232** | **1.03680** | **0.98156** |

The column arms buy their gain almost entirely on away wins and give a little
back on home wins — the same trade every rating arm in this project has made.

**The blend is the first thing that moves the draw.** 1.38950 → 1.37232, the
largest movement in that column anything here has produced, and its marginal
P(draw) of 0.2483 against an actual 0.2511 is the closest any arm has come
(v3: 0.2446).

That is worth being precise about, because it is *not* the hypothesis this
phase came to test. The score model picks draws **worse** than v3 standalone
(+0.0113 separation against +0.0119). Pooling it with v3 improves draws anyway,
because the gain comes from two forecasters *disagreeing* — a log pool is
sharper than either input where they agree and hedges where they do not, and
draws are where a 1X2 classifier is least certain. The draw improved despite
the mechanism, not because of it.

---

## Open items

1. **The blend is the first promotion candidate since the ratings phase.**
   +0.00341 [+0.00227, +0.00457], 16% of the market gap, robust across arms,
   and it clears the promotion bar in `PLAYER_STATS_SPEC.md` §6. It is also
   the cheapest structural change this project has found: the served model is
   untouched, the blend wraps it, and the serving cost is a weekly
   `fit_league` per league — four seconds of CPU for all five, against a
   feature pipeline that takes minutes. What it needs before shipping is a
   place to keep the fitted parameters between refits and a decision on where
   the weight lives, neither of which is a modelling question.

2. **The placebo cleared zero twice now, and that is the real lead.**
   `FINDINGS_LEAGUE.md`'s random interactions (+0.00074) and this phase's
   shuffled rates (+0.00032) are two unrelated noise injections that both
   improved v3 significantly. `FINDINGS_MARKET.md` open item 3 demoted the
   hyperparameter sweep on the evidence that the compression is not a
   calibration constant; that argument still holds, but it was about
   *temperature*, and this is about capacity. A depth and `REG_LAMBDA` sweep
   on the rated feature set is now the cheapest item on the board with two
   independent pieces of evidence behind it. `tune.py`'s grid was selected on
   v2 columns before `atkdef` existed.

3. **Dixon-Coles as a shape is closed.** The τ correction is worth +0.00085
   standalone, zero to the served model (`dc_probs` against `dc_rates` is a
   null), and the draw mechanism failed its pre-registered gate in both the
   online and the batch fit. The 35.5% of the market gap sitting on draws is
   still unexplained, and the one structural idea the project had for it did
   not work. What paid was the time-weighted likelihood, which is the other
   half of the same paper.

4. **Staleness is a live parameter nobody has measured for v3 itself.** The
   cadence sweep cost 0.0032 between a 7-day and a 90-day refit of a
   *two-parameter* model. v3 is refitted once per fold here and on whatever
   schedule the pipeline runs in production, and no phase has asked what that
   costs. Given how cheap the answer was to obtain for the score model, this
   is a strong candidate for the next `--diagnose`-shaped probe.

5. **The blend weight declines across folds** — 0.58, 0.75, 0.60, 0.53, 0.38.
   The score model is worth progressively less as the training set grows,
   which is what one would expect if v3 needs data and a 24-parameter league
   fit does not. If that trend continues the blend's value decays as the
   archive extends, and the weight should be refitted rather than frozen at
   whatever a `train_v3.py` measures once.

6. **Two rates matching sixty-three columns deserves its own phase.**
   `dc_swap` — the form table with `atkdef` replaced by the score model, 61
   columns — is a null against v3 at 63. Between that and the standalone
   parity result, the hypothesis that most of the form table is redundant is
   now well-supported and has never been tested directly. The phase that drops
   the 54 form columns and keeps the rates would be cheap, and after five
   consecutive feature nulls it is the question the evidence keeps pointing at.

7. **`goals_probs_rolling_dc_h365_c7.npz` caches probabilities**, per
   `FINDINGS_MARKET.md` open item 5, which is what made the by-outcome table
   and the nine-arm blend replay free.
