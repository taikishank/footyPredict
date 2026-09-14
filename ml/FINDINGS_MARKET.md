# Findings: the market benchmark

Every ablation this project has run measures an arm against the arm before
it. That answers "is this change worth shipping" and cannot answer "is there
anything left worth chasing". The ratings phase's +0.0076 reads as enormous
against the regularization sweep's +0.0044 and as rounding error against a
model that trails the closing line by 0.03 — and until now nobody knew which.

This phase gets the number.

Reproduce with:

```
./inference-py/bin/python ml/build_market_index.py --download --diagnose  # ~1 min
./inference-py/bin/python ml/rating_ablate.py --mode rolling              # ~17 min, cached
./inference-py/bin/python ml/market_benchmark.py
./inference-py/bin/python -m pytest ml/test_market_lib.py
```

`market_benchmark.py` fits nothing. It reads `rating_losses_rolling.npz`, so
the model figures here are byte-identical to the ones in
`FINDINGS_RATINGS.md` rather than a re-fit that might differ.

---

## Headline

**The gap to the closing line is 0.0211 — about 2.8× everything the ratings
phase bought.** There is real room left.

18,793 validation fixtures (94.9% of the rolling-origin set) with Pinnacle
closing prices, de-vigged:

| forecaster | log loss | share of the market's gap over no-information |
|---|---|---|
| class priors (no information) | 1.07067 | 0% |
| v2 (served today, 56 columns) | 0.99178 | 73.2% |
| + plain Elo | 0.98566 | 78.9% |
| **+ attack/defence (the best arm)** | **0.98402** | **80.4%** |
| **Pinnacle closing, de-vigged** | **0.96293** | **100%** |

Paired bootstrap, model against market, on identical fixtures:

| model | vs market | 95% CI | P(model better) |
|---|---|---|---|
| v2 | −0.02886 | [−0.03215, −0.02575] | 0.000 |
| atkdef | −0.02109 | [−0.02387, −0.01828] | 0.000 |

So the entire ratings phase moved the model from 73.2% to 80.4% of the way to
the market — 7.2 points of the 26.8 that were open. **19.6 points remain.**

This resets the scale every other finding is read on. Against the 0.0211
still available:

| change | gain | share of the remaining gap it would have closed |
|---|---|---|
| attack/defence ratings | +0.0076 | 36% |
| Phase 9's availability block | +0.0061 | 29% |
| `club_goals_relative` | +0.0054 | 26% |
| the entire regularization sweep | +0.0044 | 21% |

Three or four more findings the size of the ones already banked would reach
the closing line. That is a meaningful target rather than a hopeless one —
but it is also not one more feature away.

---

## The gap is information, not calibration

This is the finding that decides what to build next, and it rules out the
cheap fixes.

**Marginal calibration is fine.** Over the same 18,793 fixtures:

| | home | draw | away |
|---|---|---|---|
| actual | 0.4412 | 0.2511 | 0.3078 |
| model (atkdef) | 0.4522 | 0.2444 | 0.3033 |
| market | 0.4440 | 0.2486 | 0.3074 |

A slight home lean, nothing that explains 0.021.

**But the model's probabilities are compressed toward the base rate.**
Bucketing by the market's P(away) into deciles — the market's ranking, not the
model's, so the buckets aren't sorted by the model's own errors:

| market P(away) decile | n | model says | market says | actually happened |
|---|---|---|---|---|
| 0.011–0.102 | 1880 | 0.1041 | 0.0671 | 0.0676 |
| 0.102–0.153 | 1879 | 0.1512 | 0.1284 | 0.1213 |
| 0.278–0.321 | 1880 | 0.2953 | 0.2994 | 0.2947 |
| 0.447–0.568 | 1879 | 0.4754 | 0.5025 | 0.5104 |
| 0.568–0.909 | 1879 | 0.5872 | 0.6674 | **0.6881** |

The market is almost exactly right at both ends (0.0671 against 0.0676;
0.6674 against 0.6881). The model is 3.7 points too high at the bottom and
**10.1 points too low at the top**. Separation confirms it: mean P(away) on
fixtures that ended in an away win minus fixtures that didn't is +0.1204 for
the model against +0.1501 for the market. The market discriminates about 25%
better.

**And sharpening does not fix it.** Temperature scaling, fitted on the first
half of the validation rows and applied to the second:

```
T = 0.945 (a mild sharpen)
held out:  raw 0.98674  ->  tempered 0.98713   (-0.00039, i.e. worse)
```

A monotone transform of the model's own output recovers *nothing* — it costs
0.0004. So the compression is not a global over-confidence setting that a
recalibration layer could undo. When the market says 0.68 the model says 0.59
because **it does not identify those fixtures as extreme in the first place.**
The missing thing is signal, not a scaling constant.

This kills two shortcuts that would otherwise look attractive: a calibration
layer bolted onto v2, and the assumption behind open item 3 in
`FINDINGS_RATINGS.md` that the v2 hyperparameters are simply over-regularised
for the rated feature set. Re-tuning `REG_LAMBDA` is still worth doing, but
it is now clearly a small-change item, not the explanation.

---

## Where the gap sits

By actual outcome:

| outcome | n | model | market | gap | share of total gap |
|---|---|---|---|---|---|
| home_win | 8291 | 0.70754 | 0.70623 | +0.00130 | **2.7%** |
| draw | 4718 | 1.39053 | 1.36073 | +0.02981 | 35.5% |
| away_win | 5784 | 1.04875 | 1.00639 | +0.04236 | **61.8%** |

**On fixtures the home side wins, the model is level with the closing line.**
Essentially the entire deficit is on fixtures where the home side does not
win — and away wins alone are 62% of it.

By how one-sided the market thinks the fixture is, the gap is close to flat
(16.6% / 23.8% / 24.8% / 20.9% / 13.9% across five buckets, with the widest
per-fixture gap on heavy favourites). So this is not a mismatch confined to
one kind of fixture; it is general, and it shows up as an inability to be
confident about away sides.

That points somewhere specific. The 54 form columns are split home/away and
are opponent-blind; `atkdef` fixed the opponent-blindness at the *club* level
but still enters as a single pre-match expectation. Nothing in the feature
set describes a club's **away-specific** strength — a side that travels badly
and one that doesn't produce the same vector. That is the same shape of
omission opponent strength was before the ratings phase, and it is now the
best-evidenced hypothesis on the board.

The draw's 35.5% is the other third, and it connects to open item 5 in
`FINDINGS_RATINGS.md`: the market's P(draw) separation is +0.0176 against the
model's +0.0117, so the market is *also* barely able to pick draws — it is
just less bad at it. Draws may be where the least is available per unit of
effort.

---

## What the odds actually cost: nothing

The obvious source was wrong, and checking that first saved a subscription.

**The Odds API (what `ingestor-go/internal/odds` already uses) was priced and
rejected.** It sells history, but: its archive starts 2020-06-06, which
reaches only 56.6% of the validation set; historical snapshots cost 10 quota
credits each against 1 for a live call; and one snapshot per league per
match-day over that span is 3,686 calls — **~36,900 credits**, which is the
$59/month tier. The free tier returns an explicit refusal:

```
HTTP 401  HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN
```

**football-data.co.uk is free and strictly better here.** Same five leagues,
CSVs back to the 1990s, no key and no quota, and it carries Pinnacle's
*closing* prices — the sharper benchmark, since closing is the price after
team news. Coverage of the validation span:

| price source | rows | % | mean overround |
|---|---|---|---|
| Pinnacle closing | 18,907 | 95.7% | 2.54% |
| Pinnacle opening | 18,893 | 95.6% | 2.75% |
| market average closing | 12,459 | 63.0% | 4.73% |
| Bet365 opening | 19,757 | 100.0% | 5.19% |

96% coverage at $0, against 57% at $59/month.

**The de-vig choice is not load-bearing.** Both methods are implemented
(`market_lib.py`) because the proportional one is known to be biased —
bookmakers load proportionally more margin onto longshots, which would
understate favourites and land exactly on the draw. Shin's method corrects
for it. On these prices the two differ by **0.00016**, against a 0.021 gap, so
the headline survives either choice. Worth having measured rather than
assumed; at Bet365's 5.19% margin the two would have been further apart.

---

## Why the join is learned rather than hand-written

The two sources spell clubs differently — `West Bromwich Albion` / `West
Brom`, `FC Bayern München` / `Bayern Munich`, `Olympique Lyonnais` / `Lyon`.

Porting the ingestor's approach (`internal/odds/match.go`: token-overlap plus
a hand-maintained alias table) reached **79.2%** of fixtures. That is the
right design *live*, where a never-before-seen event has to be matched right
now. Offline it is the wrong one, because the 21% it misses are not random:
they are the clubs whose two spellings share no tokens, which means the same
clubs, in the same leagues, in every season they appear. A name-matching gap
would have become a systematically missing subset of the benchmark — the
exact failure mode `--diagnose` exists to catch.

The archive offers something the live path does not: **both sources already
know the answer.** League, date and scoreline pin a fixture down without
consulting a name at all. So the match runs in two passes — first on
(league, date ±1, scoreline) where that identifies exactly one candidate,
consulting no names; then harvesting what those matches imply each club is
called on the other side, and re-matching everything on the learned names
with no scoreline condition (so a fixture whose scoreline was ambiguous, or
whose sources disagree on the score, still matches).

Result: **164 club names learned, 0 ambiguous, 99.1% of fixtures matched**,
with 99.985% scoreline agreement on the matches made. A club's name is only
accepted if it takes ≥70% of that club's votes; a split vote is reported
rather than resolved by plurality, since a split means the first pass paired
something wrongly and taking the plurality would bake that error into every
season the club appears in.

The 175 unmatched fixtures are 174 from the 2026/27 season (beyond the
downloaded season files) and one 2023 playoff.

---

## One diagnostic flag, deliberately not fixed

`--diagnose` correlates every emitted column against the fixture date, the
check that caught three bugs in the ratings phase. Every *probability* column
is clean (|r| < 0.04). The `overround` columns are not:

```
pinnacle_closing_overround   +0.542
bet365_open_overround        +0.356
```

Bookmaker margins in this archive widen over time. That is a real property of
the source and possibly worth understanding, but the overround is a
diagnostic column — it is never scored and never fed to a model — so it is
recorded here rather than corrected. It would matter if anyone used margin as
a feature.

---

## Open items

1. **Away-specific team strength is now the best-evidenced hypothesis.**
   61.8% of the gap is on away wins and the model is level with the market on
   home wins. Nothing in the feature set distinguishes a club's home form from
   its away form; `atkdef` carries one home-advantage parameter *per league*,
   not per club. Splitting the ratings into home and away attack/defence, or
   adding venue-split form, is the direct test.

2. **The form-window sweep is done, and it bought nothing** —
   `FINDINGS_WINDOW.md`. The +0.002 guessed at here was optimistic by a factor
   of five: the best of eleven window arms is +0.0004, 1.9% of this gap, with
   an interval touching zero, and both shortening and lengthening the window
   are significantly *worse*. Opponent-adjusted form statistics are untouched
   and remain an open item; they are a different fix from a window length.

3. **Re-tuning `REG_LAMBDA` on the rated feature set is demoted.** The
   temperature result shows the compression is not a regularisation setting.
   Still worth the cheap sweep; no longer a candidate explanation.

4. **The benchmark is a lower bound on what is achievable, not an upper one.**
   Pinnacle's closing line incorporates team news, lineups and money the model
   never sees. Matching it is not the ceiling of what is *possible*; it is the
   ceiling of what is possible without that information. Phase 9's lineup
   backfill is the only part of this project that touches it.

5. **`rating_losses_rolling.npz` caches losses, not probabilities.** Every
   calibration number above needed a 5-fold re-fit of one arm to recover the
   probabilities. Caching `predict_proba` output instead would make
   calibration analysis free, at the cost of a one-off 17-minute re-run and a
   larger cache.

6. **The market is not in `backtest/report.py`'s terms.** That path already
   computes a model-vs-market Brier on live graded fixtures, over a few weeks
   of rows. It and this benchmark should agree in direction; nobody has
   checked, and a disagreement would be informative about the train/serve skew
   in open item 4 of `FINDINGS_RATINGS.md`.
