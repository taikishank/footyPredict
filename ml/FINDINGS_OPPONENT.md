# Findings: opponent-adjusting the form window

`feature_lib.team_form()` averages a club's last `MAX_WINDOW = 8` matches and
is opponent-blind: eight matches against relegation sides and eight against
the top four produce the same 27-field vector. The model knows exactly who a
club is *about* to play - `atkdef`'s four rating columns say so - and nothing
at all about who it has just played.

`FINDINGS_VENUE.md` open item 5 names this as the last remaining *feature*
hypothesis with a mechanism behind it, and in the same sentence warns about
it: `club_goals_relative` was absorbed by `atkdef`, and opponent-adjusted form
may be too.

## Pre-registration

Written before the probe was run, and before any of its output was read.
`FINDINGS_VENUE.md` is the precedent: that phase learned in two minutes of
`--diagnose` what its 35-minute ablation would have taken an afternoon to
confirm, and this phase inherits the discipline rather than the conclusion.

**The staged plan.** Cheapest arm first. `opp_faced` adds two columns per club
- the mean attack and mean defence rating of the opponents the form window
actually saw - and nothing else. Only if that arm pays does the phase escalate
to per-stat residual adjustment (`resid_scalar`), then to per-stat allowed
ratings (`resid_allowed`), each with the shuffled-opponent placebo
`FINDINGS_VENUE.md` built. A 54-column apparatus that cannot beat two columns
has not earned its parameters.

**The bar the diagnose has to clear for the ablation to run at all.** Both,
together:

1. **Headroom.** For at least 5 of the 27 form fields, the share of the form
   column's variance attributable to the schedule - `Var(b·opp_faced)` over
   `Var(form_mean)`, with `b` fitted per field on per-match values - is at
   least **5%**. Below that the correction cannot move a column far enough to
   matter whatever its sign.

2. **Reliability.** Those same fields show a **positive out-of-sample R²
   gain** when `opp_faced` is added to a regression predicting the club's own
   *next-match* value for that field, on a chronological holdout, where both
   models already carry the strength of the opponent the club is about to
   face. This is the ablation in miniature: v3 already knows the next
   opponent, so the only thing on trial is knowledge of the schedule behind
   the window.

3. **Non-redundancy.** At least one of the qualifying fields is outside
   `{goals_for, goals_against}`. `atkdef` is a Poisson-fitted,
   opponent-adjusted goals rating already sitting in v3; a headroom result
   confined to the two goal fields is a restatement of a column the model has.

If the bar is not cleared, the phase stops at the diagnose and the ablation is
not run.

---

## Headline

**The bar was not cleared, so the ablation was not run.** Zero of 27 fields
reach a 5% schedule share; the largest is `goals_against` at 3.9% and the
largest outside the two fields `atkdef` already covers is `possession` at 1.5%.

Reproduce with:

```
./inference-py/bin/python ml/build_opponent_features.py --diagnose   # ~1 min
./inference-py/bin/python -m pytest ml/test_form_lib.py
```

`ml/form_lib.py` and `ml/build_opponent_features.py` are new. `feature_lib.py`
is untouched, nothing in the serving path changed, and no model was fitted.

## The mechanism is real. It does not survive the average.

This is the first phase here whose hypothesis is confirmed at the level it was
stated and dead one level up. An opponent does measurably change what a club
produces, in the right direction every time:

| field | per-match R² on the opponent played | b(opp_atk) | b(opp_def) |
|---|---|---|---|
| possession | 0.0896 | −12.54 | −5.95 |
| goals_against | 0.0749 | +1.40 | +0.52 |
| passes | 0.0399 | −79.66 | −33.09 |
| shots_total | 0.0347 | −1.78 | −3.03 |
| goals_for | 0.0321 | −0.48 | −0.95 |

A better defence takes away your possession, passes, shots and goals; a better
attack raises what you concede and the saves you make. Nine per cent of the
match-to-match variation in possession is *who you played*. If the adjustment
were applied to a single match it would be worth having.

It is not applied to a single match. It is applied to a mean of eight, and the
opponent term is averaged along with everything else:

| | per match | over 8 | shuffled null | 95% of null |
|---|---|---|---|---|
| `opp_atk` | 0.1994 | **0.0589** | 0.0739 | [0.0733, 0.0748] |
| `opp_def` | 0.1563 | **0.0458** | 0.0578 | [0.0571, 0.0584] |

Opponent strength varies by 0.20 from match to match and by 0.06 across form
windows. **And the observed spread is below the shuffled null, outside its
interval** — a club's last eight opponents are *more even* than eight drawn at
random from the same league-season, at about 80% of the random spread on both
ratings.

That is not a curiosity, it is the mechanism. A league is a round robin:
having already played the strong sides is exactly what leaves the weak ones to
come, so the draw is self-correcting on any horizon short of a season. There
is no club-level schedule trait for an adjustment to recover, and less left
over from chance than chance alone would have given it.

`FINDINGS_VENUE.md` killed club venue strength with the same statistic in the
same shape — an observed club-to-club spread smaller than sampling noise
alone would produce. Two phases, two hypotheses, one reason.

## What the correction would have been worth, in units

The clearest way to read it. `correction` is the standard deviation of the
whole opponent adjustment for a field; `club spread` is the standard deviation
of the form column it would be subtracted from:

| field | correction | club spread | ratio | schedule share |
|---|---|---|---|---|
| possession | 0.904 | 7.303 | 12.4% | 1.5% |
| goals_against | 0.096 | 0.485 | 19.9% | 3.9% |
| shots_total | 0.211 | 2.103 | 10.0% | 1.0% |
| passes | 5.593 | 72.608 | 7.7% | 0.6% |
| goals_for | 0.062 | 0.556 | 11.2% | 1.3% |
| corners | 0.103 | 1.057 | 9.8% | 1.0% |

The entire opponent adjustment for possession is **0.9 percentage points**,
against a club-to-club spread of 7.3. For shots, a fifth of a shot against
two shots. Because variance goes as the square, a correction one-tenth the
size of the column moves one per cent of its variance — the ratio column
looks survivable and the share column is what the model would actually see.

## The reliability half agrees, and it is the half that was binding

Predicting a club's own next-match value for each field, on a chronological
holdout, where both models already carry the strength of the opponent about to
be faced — which is what v3 has — and only the schedule behind the window is
added:

| field | ΔR² |
|---|---|
| possession | +0.00164 |
| goals_for | +0.00094 |
| shots_total | +0.00082 |
| fouls | +0.00074 |
| goals_against | +0.00025 |
| penalties, tackles, saves, red cards | negative |

The gains are positive far more often than not, which says the direction is
right and the probe is not noise. They are also two to three orders of
magnitude below anything that has ever moved this model's log loss. The
best field in the table adds 0.0016 of R² to predicting *possession* — an
intermediate quantity, not a result.

Note which fields fail outright: `saves` at −0.00007 despite a 2.6% per-match
R² and a correctly-signed coefficient. Estimating a correction costs variance,
and where the correction is small enough the estimate is worse than leaving
the column alone. That is the `placebo` result from `FINDINGS_VENUE.md`
arriving before the ablation rather than inside it.

## Why the staged plan mattered

The plan was `opp_faced` (two columns per club) first, escalating to per-stat
residual adjustment and then to 27 per-stat allowed ratings only on a positive.
None of those escalations would have fixed anything: every one of them is a
better *estimator* of a quantity whose spread is 0.06 where the thing it
corrects has a spread of 7.3. `resid_allowed` would have carried 54 parameters
to chase a correction smaller than its own estimation error, and the placebo
arm would very likely have beaten it, at a cost of about forty minutes of
fitting to learn what one minute of arithmetic says here.

This is the same lesson `FINDINGS_VENUE.md` drew and the discipline is now
worth stating as a rule: **a quantity's spread across the rows the model sees
bounds what any feature built on it can be worth, and that bound is cheap to
compute.** Per-match R² is not that bound. Neither is coefficient significance.

## Open items

1. **Opponent-adjusted form is closed.** `FINDINGS_VENUE.md` open item 5
   called it the last remaining feature hypothesis with a mechanism behind it.
   The mechanism is confirmed and it is too small to reach the form window.
   Neither the residual nor the allowed-ratings escalation is worth running.

2. **`FINDINGS_MARKET.md`'s away-win gap survives another phase.** It is now
   unexplained by club venue strength, by a stale home-advantage constant, by
   the form horizon, and by opponent context. Every feature hypothesis this
   project has generated from the form table has come back null.

3. **What is left is not features.** The train/serve skew
   (`FINDINGS_RATINGS.md` open item 4) is a correctness item — the offline
   number is wrong where it counts and no amount of it is a gain. The
   Dixon–Coles goals model is the only remaining item that changes the
   *shape* of the model rather than its inputs, and `AttackDefence` is already
   the parameterisation it needs. Given four consecutive feature nulls, the
   honest read is that the form table is exhausted and the next real move is
   structural.

4. **`opponent_faced.csv` exists and is joinable** (39,619 rows, `fixture_id`,
   four columns) if a later phase wants the schedule columns for something
   else. Nothing reads it today.

5. **The cheap-bound probe should come first from now on.** Three phases have
   now spent their compute confirming what a variance ratio said in advance.
   A `--diagnose` that bounds the hypothesis before the ablation is fitted is
   the highest-return habit this project has.
