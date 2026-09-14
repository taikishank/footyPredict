# Findings: home advantage, at the club and at the league

`FINDINGS_MARKET.md` put 61.8% of the gap to Pinnacle's closing line on away
wins, found the model level with the market on home wins, and named the fix in
its open item 1: nothing in the feature set distinguishes a club that travels
badly from one that doesn't. That was the best-evidenced hypothesis the
project had.

**It was the wrong hypothesis, and the right one pays nothing.** A club-level
home advantage does not exist in this archive - a club's venue gap does not
predict its own other half-season. League home advantage *has* fallen, the
served rating was carrying a stale one, and fixing it corrects the home lean
almost exactly while buying +0.00035 with an interval through zero.

Reproduce with:

```
./inference-py/bin/python ml/build_venue_features.py --diagnose   # ~2 min
./inference-py/bin/python ml/venue_ablate.py --mode split         # ~3 min
./inference-py/bin/python ml/venue_ablate.py --mode rolling       # ~35 min
./inference-py/bin/python -m pytest ml/test_venue_lib.py
```

`ml/venue_lib.py` and `ml/build_venue_features.py` are new; nothing in the
serving path changed.

---

## Headline

Rolling-origin validation, 19,797 fixtures over five expanding-window folds.
Every arm is a drop-in replacement for the served `atkdef` rating, so every
arm carries exactly **63 columns** and differs from v3 in the rating rule and
nothing else:

| arm | log loss | vs. v3 | 95% CI | verdict |
|---|---|---|---|---|
| **v3 (served)** | **0.98497** | — | — | — |
| `hfa_served` (AdaptiveHFA at the served rate) | 0.98497 | +0.00000 | [+0.00000, +0.00000] | integrity check |
| `hfa_lr10` (home-advantage rate ×10) | 0.98461 | +0.00035 | [−0.00025, +0.00097] | null |
| `hfa_lr100` (×100) | 0.98513 | −0.00017 | [−0.00105, +0.00078] | null |
| `hfa_roll` (380-fixture rolling estimate) | 0.98513 | −0.00017 | [−0.00086, +0.00061] | null |
| `venue` (four ratings per club, split by venue) | 0.98598 | −0.00102 | [−0.00226, +0.00024] | null |
| `placebo` (**random** split, same shape) | 0.98739 | −0.00242 | [−0.00374, −0.00117] | **worse** |

On `FINDINGS_MARKET.md`'s scale, the gap to the closing line is 0.0211. The
best arm here is 1.7% of it and does not clear its own interval.

---

## The club trait was dead before the ablation started

`FINDINGS_LEAGUE.md` established the shape of this check: a quantity that does
not survive out of sample cannot be worth a column however large its in-sample
spread. Here it is again, and it is more decisive than it was for leagues.

2,152 club-seasons with at least 12 matches at each venue, 210 clubs:

```
league-mean venue gap                          0.482 ppg
club spread within league-season         sd =  0.373 ppg
spread expected from sampling noise alone =    0.431 ppg   (19 matches per venue)
```

**The observed club-to-club spread is smaller than what pure scoreline noise
would produce.** There is no room left in it for a real trait.

Persistence agrees, twice over:

| | r | shuffle null, 95% |
|---|---|---|
| lag-1, venue gap in points | +0.0359 | [−0.0459, +0.0466] |
| lag-1, venue gap in goal difference | +0.0165 | [−0.0464, +0.0461] |
| **within-season split-half** | **−0.0314** | — |

The split-half number is the one that settles it. A club's venue gap measured
on half its matches does not predict its own other half, so the lag-1
correlation has a ceiling of zero and cannot be raised by a better estimator,
a longer window, or a smarter rating. "A side that travels badly and one that
doesn't" produce the same vector because, in the results, they are the same
side.

## The placebo is what makes the venue arm's null readable

`venue` gives each club four ratings - attack and defence, home and away -
each fed only the matches at its own venue. `placebo` has the identical shape,
with the two slots assigned by fixture id instead of by venue: same parameter
count, same halving of the evidence behind each rating, no football in the
split.

```
placebo   -0.00242   [-0.00374, -0.00117]    splitting a club's ratings costs this
venue     -0.00102   [-0.00226, +0.00024]
venue vs placebo   +0.00141  [-0.00007, +0.00282]   P(venue better) = 0.969
```

So halving the evidence costs 0.0024, the venue split recovers about 0.0014 of
that, and the recovery does not clear its interval. Read without the control,
`venue`'s −0.00102 looks like "venue helps a little but not enough to pay for
itself". Read with it, the honest statement is that **the venue split is not
distinguishable from a random split of the same shape, and both are worse than
not splitting at all.** FINDINGS_LEAGUE.md's `interact` arm is the precedent
this control exists for.

---

## League home advantage moved, and the served rating did not follow

The probe that redirected the phase. On `AttackDefence`'s own scale - `hfa` is
a log multiplier on the home side's goal rate, so the archive's value is
log(mean home goals) − log(mean away goals):

| era | archive | carried by v3 |
|---|---|---|
| 2005–2015 | 0.3029 | ~0.21 → 0.27 (burning in) |
| 2021–2025 | 0.2011 | 0.229–0.245 |
| 2020 (empty stadiums) | 0.1187 | 0.2599 |
| 2024 | 0.1217 | 0.2402 |

Away-win rate went 0.277 → 0.311 over the same span and every league moved the
same way (league 384: 0.304 → 0.133). `AttackDefence.hfa` is seeded at
`INIT_HFA = 0.20` and updated at `lr * 0.01` = 0.0002 per fixture against a
league-wide average, so it is effectively frozen: it drifts *up* through the
first decade while the archive drifts *down* through the second.

This is a genuine bias, in the exact direction `FINDINGS_MARKET.md` measured
as a home lean, and the arms remove it:

```
                P(home)   P(draw)   P(away)
actual           0.4406    0.2511    0.3083
v3               0.4512    0.2446    0.3042
hfa_lr10         0.4467    0.2453    0.3080
hfa_roll         0.4463    0.2452    0.3085
```

By actual outcome, the loss moves precisely where the market phase said the
gap was:

| arm | home_win | draw | away_win | pooled |
|---|---|---|---|---|
| v3 | 0.71024 | 1.38950 | 1.04813 | 0.98497 |
| `hfa_lr10` | 0.72018 | 1.38616 | **1.03549** | 0.98461 |

**And that is the finding.** Away-win loss falls by 0.0126, home-win loss rises
by 0.0099, and the two nearly cancel. Removing a real, correctly-diagnosed
bias in the correct direction is worth +0.00035, which does not clear zero.

`FINDINGS_MARKET.md` said this in advance and was believed only halfway: "the
compression is not a global over-confidence setting that a recalibration layer
could undo... the missing thing is signal, not a scaling constant." A stale
home advantage is a scaling constant. Temperature scaling was one test of
that claim; this is the other, from inside the rating rather than on top of
the output, and it agrees. The away-win share of the gap is not a home/away
*bias* - it is the model failing to identify which away sides are strong, and
no correction applied equally to every fixture can fix that.

## Tracking the drift better makes it worse

The three HFA arms sit in a clear order, and it is not the order the drift
table predicts:

| arm | how closely it tracks the archive | log loss |
|---|---|---|
| `hfa_lr10` | loosely | **0.98461** |
| `hfa_lr100` | to within ~0.01 (2020: 0.1187 actual, 0.1221 carried) | 0.98513 |
| `hfa_roll` | to within ~0.05, on a 380-fixture horizon | 0.98513 |

Paired directly, `hfa_lr10` beats `hfa_roll` by +0.00052 [+0.00009, +0.00094],
P = 0.996, and `hfa_lr100` by +0.00052 [−0.00023, +0.00128].

A season's HFA estimate carries about 0.03 of noise on the log scale against a
drift of about 0.10 across two decades, so an arm that reproduces the annual
series exactly is reproducing its noise as well as its trend. The mild arm is
the one that separates them. This is the same lesson as
`FINDINGS_WINDOW.md`'s flat-bottomed curve: the best estimator of a moving
quantity is not the one that moves fastest.

## One diagnostic flag that should fire

`--diagnose` correlates every emitted column against the fixture date, the
check that caught three bugs in the ratings phase. The HFA arms flag
`exp_goals_away` at +0.16 where the served arm sits at +0.077.

That is correct behaviour, not a bug. Away scoring genuinely rose across the
archive (1.05 to 1.33 goals a game), so a rating that tracks home advantage
honestly *must* show a column that rises with the date. The flag is worth
keeping - it is doing its job by asking - but the answer here is that the
archive moved, not that the column is an era proxy. Note the flag fires in the
opposite direction too: `hfa_lr100_exp_goals_home` correlates +0.022 with the
date against the served arm's +0.106, so the adaptive arms are *less* of an
era proxy overall.

---

## Open items

1. **FINDINGS_MARKET.md open item 1 is closed.** Club-level venue strength was
   the project's best-evidenced remaining hypothesis and it is measured dead
   twice - by a reliability probe that bounds it at zero, and by an ablation
   whose placebo beats the real split's claim to significance. Away-specific
   form columns are the same hypothesis in a different table and do not need
   their own phase.

2. **The away-win share of the gap is still unexplained.** It survives the
   bias fix: `hfa_lr10` corrects the marginal rates almost exactly and away
   wins remain the worst-predicted outcome by a wide margin (1.0355 against
   0.7202 on home wins). Whatever identifies a strong away side to the closing
   line is not a venue parameter and not a calibration constant.

3. **`hfa_lr10` is a free half-tick if a v3 artifact is ever built.** +0.00035
   is null and not a reason to ship anything, but it costs one constructor
   argument, it strictly improves marginal calibration, and it removes a known
   bias that will keep growing as the archive extends. If `train_v3.py` is
   written anyway, the case for `hfa_rate=0.10` over `0.01` is "no measured
   cost, one known bias removed" rather than a log-loss gain.

4. **The probability cache exists now.** `venue_probs_rolling.npz` stores
   `predict_proba` output rather than losses, which is `FINDINGS_MARKET.md`
   open item 5. The by-outcome and marginal-rate tables above cost nothing to
   produce; the same analysis against `rating_losses_rolling.npz` needed a
   17-minute re-fit. Later phases should cache probabilities.

5. **Untouched, and now the top of the list**: the train/serve skew
   (`FINDINGS_RATINGS.md` open item 4), the Dixon–Coles goals model (open item
   5, and `AttackDefence` is still the parameterisation it needs), and
   opponent-adjusting the 54 form statistics. The third of those is the only
   remaining *feature* hypothesis with a mechanism behind it, and this phase
   is a caution about it: `club_goals_relative` was absorbed by `atkdef`, and
   opponent-adjusted form may be too.
