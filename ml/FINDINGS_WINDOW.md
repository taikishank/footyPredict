# Findings: the form window

`MAX_WINDOW = 8` is the oldest unmeasured constant in the project. Every form
column averages a club's last eight matches, and nothing chose eight —
it predates every ablation here. `FINDINGS_PHASE9.md` §5 flagged it as the
cheapest promising experiment left, on the strength of `club_goals_relative`:
club scoring over **38** matches, worth +0.0035 there, with raw and normalised
forms worth the same, which says the horizon carried it rather than the
normalisation.

This phase swept the window on top of the v3 recipe (v2 form +
`atkdef` ratings, `FINDINGS_RATINGS.md`). The answer is that **eight is
already right**, and the premise behind the hypothesis no longer holds.

Reproduce with:

```
./inference-py/bin/python ml/build_window_features.py       # ~8 min, once
./inference-py/bin/python ml/window_ablate.py --mode split  # ~2 min
./inference-py/bin/python ml/window_ablate.py --mode rolling   # ~35 min
./inference-py/bin/python -m pytest ml/test_window_ablate.py
```

`ml/feature_lib.py` gained one default argument (`team_form(history,
window_size=MAX_WINDOW)`) so one replay can emit every window. Training and
inference both take the default; the served recipe is unchanged, and
`build_window_features.py` asserts its w8 table reproduces
`2000_window3_8.csv` column-for-column on all 39,594 fixtures the two share.

---

## Headline

Rolling-origin validation, 19,810 fixtures over five expanding-window folds,
every arm carrying the same `atkdef` rating columns and the same v2
hyperparameters:

| arm | log loss | vs. w8 | 95% CI | verdict |
|---|---|---|---|---|
| w4 | 0.98663 | −0.00142 | [−0.00223, −0.00058] | **worse** |
| w6 | 0.98568 | −0.00046 | [−0.00111, +0.00016] | null |
| **w8 (served)** | **0.98521** | — | — | — |
| w12 | 0.98480 | +0.00042 | [−0.00024, +0.00108] | null |
| w16 | 0.98495 | +0.00026 | [−0.00054, +0.00104] | null |
| w24 | 0.98543 | −0.00021 | [−0.00117, +0.00073] | null |
| w38 | 0.98628 | −0.00106 | [−0.00209, −0.00008] | **worse** |
| g38_s8 (goals over 38, stats over 8) | 0.98482 | +0.00040 | [+0.00000, +0.00079] | borderline |
| g16_s8 | 0.98533 | −0.00011 | [−0.00038, +0.00015] | null |

The loss curve is flat-bottomed between 6 and 24 and rises at both ends. Both
ends are *significantly* worse: cutting to four matches costs −0.00142,
stretching to 38 costs −0.00106. Nothing inside the flat region clears zero.

The w8 arm scoring 0.98521 against `FINDINGS_RATINGS.md`'s 0.98503 for the
same recipe is the cross-check that this is the same experiment: the 0.00018
difference is 25 fixtures the archive gained since that table was built.

**On the scale that matters**, from `FINDINGS_MARKET.md`: the gap to
Pinnacle's closing line is 0.0211. The best arm here is +0.0004 — **1.9% of
the remaining gap**, and it does not clear its own interval. Item 2 of that
document's open list guessed a window sweep might buy +0.002 (9.5% of the
gap); it buys a fifth of that, and cannot be distinguished from zero.

---

## The premise died on the way

The reason to expect a longer window was `club_goals_relative`: a 38-match
goal horizon worth +0.0054 as the v1 → v2 feature, and the most important
single column in Phase 9's challenger. Two arms drop it, to ask whether a
long form window is only rediscovering what that column already carries.

| | log loss | `club_goals_relative` worth |
|---|---|---|
| w8 with it | 0.98521 | +0.00018 [−0.00011, +0.00046] |
| w8 without it (61 cols) | 0.98540 | |
| w38 with it | 0.98628 | +0.00011 [−0.00011, +0.00031] |
| w38 without it (61 cols) | 0.98639 | |

**`club_goals_relative` is now worth nothing.** +0.00018 with an interval
through zero, against +0.0054 when it was added. The `atkdef` ratings absorbed
it — which is unsurprising in hindsight, since a Poisson-fitted attack rating
*is* long-horizon scoring rate, with the opponent divided out and no 38-match
cutoff at all.

That is the real result of this phase, and it settles the window question
rather than answering it: the evidence that the horizon mattered came from a
column whose value has since been consumed by a better-shaped feature. There
was no long-horizon signal left for a longer window to pick up.

It also means the served v3 recipe can drop two columns for −0.0002, which is
not a reason to drop them (the interval spans zero in both directions) but is
worth knowing before anyone defends them.

## The one arm that isn't flat

`g38_s8` — goals over 38 matches, match statistics over 8 — is the only arm
with a positive point estimate in **both** modes (+0.00040 rolling, +0.00014
on the single split). Its rolling interval is [+0.00000, +0.00079].

It should not be promoted, for three reasons:

1. The lower bound is zero to five decimal places. That is the definition of
   the bar this project set, not a pass of it.
2. It is the best of eleven arms. The best of eleven beats a baseline by some
   margin even when all eleven are worthless — the same argument
   `rating_ablate.py`'s docstring makes about testing arms on one split.
3. It is +0.0004 against a 0.0211 gap, and it would cost the serving path a
   second, longer history query per club.

It is recorded because the *shape* is informative, not the size: the only
direction with any life in it separates goals from match statistics, which is
consistent with everything above — scoring rate wants a long horizon, and
possession and card counts do not.

---

## Method

`ml/build_window_features.py` replays the archive once and emits one table per
window in `data/features/window_sweep/`. Three choices worth recording:

**The `MIN_WINDOW = 3` gate is held fixed at every window.** A fixture is in
every table or in none (39,619 rows each), so arms pair fixture-for-fixture
and the bootstrap resamples differences. Letting the gate track the window
would have confounded "longer horizon" with "different, more-established
fixtures" — and would have deleted exactly the young-club rows a long window
is supposed to help.

**A short window is not a truncation of a long one.** `team_form()` averages
what exists, so a club with five matches of history gets the same five-match
mean under w8 and w38. The tables differ only where a club has more history
than the shorter window admits, which is why the w4 → w38 spread is as narrow
as it is.

**Arms compose by column selection.** Every table carries identical column
names, suffixed `__w<n>` at load, so `g38_s8` needed no extra build — it is
w38's two goal columns beside w8's twenty-five statistic columns.
`ml/test_window_ablate.py` tests that composition, since an arm silently
taking its goals from the wrong window would read as a finding.

Everything downstream is `rating_ablate.py`'s machinery, imported rather than
restated: the same five expanding-window folds, the same paired bootstrap on
per-fixture losses, the same `train_v2` hyperparameters.

---

## Open items

1. **The hyperparameters were tuned at window 8** (`tune.py:259`). A longer
   window is a smoother, lower-variance feature and might prefer a different
   depth or learning rate, so this is a null result *at this tuning*. The
   honest form of the caveat: a window sweep and a hyperparameter sweep are
   not separable, and running both jointly is 11 × the grid. Given the sweep's
   ceiling is 1.9% of the remaining gap, that is not worth buying.

2. **This closes item 2 of `FINDINGS_MARKET.md`'s open list.** The window is
   not where the 0.0211 is. Away-specific team strength (item 1 there) is
   untouched and remains the best-evidenced hypothesis — and note that the
   arms here all failed in a way that is consistent with it: averaging *more*
   matches cannot separate a club's home strength from its away strength, so
   no window length was ever going to reach that signal.

3. **Opponent-adjusted form statistics are still untested.** The ratings
   adjust the *result* for opponent strength; the 25 statistic columns are
   still raw per-match averages against whoever happened to be playing. That
   is a different fix from a window length, and this phase says nothing
   about it.

4. **`window_losses_rolling.npz` caches losses, not probabilities** — the same
   limitation `FINDINGS_MARKET.md` item 5 records. Any calibration question
   about these arms needs a re-fit.
