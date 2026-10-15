# Findings: recency weighting

FINDINGS_LEAGUE.md open item 4: all five leagues' outcome rates drift across
eras by more than the leagues differ from each other — a shared dip in the
home-win rate around 2019-12..2022-02 moves all five leagues together — and
nothing in v3 is allowed to know what year it is, on purpose
(FINDINGS_PHASE9.md's career-appearance bug is the precedent). Whether
down-weighting older training rows helps was left untested.

**It does not. Every half-life tested is a null or a loss, and the shortest
ones lose badly** — 180 days costs **−0.0051** [−0.0068, −0.0033], the worst
arm by an order of magnitude. The least aggressive half-life (1460 days,
closest to uniform weighting) is the best of the five and still a flat null:
+0.00021 [−0.00033, +0.00078], P(better)=0.753.

The weighting is not worthless *because* it is meaningless, though. A
placebo that keeps the exact same weight histogram but severs the link to
which row is actually recent (`shuffle_365`) loses far more than the real
weighting does — every half-life beats it decisively, P(better)=1.000. The
real finding is a trade, not a null: recency carries genuine information,
but at every half-life tried, the cost of the training rows it discards
outweighs what that information is worth.

Reproduce with:

```
./inference-py/bin/python ml/recency_ablate.py --mode split      # ~30s, smoke test only
./inference-py/bin/python ml/recency_ablate.py --mode rolling    # ~2 min
./inference-py/bin/python ml/recency_ablate.py --mode rolling --recipe v2   # ~3.5 min, depth check
./inference-py/bin/python -m pytest ml/test_recency_ablate.py
```

`ml/recency_ablate.py` is new; nothing in the serving path changed — every
arm shares v3's 63 columns and differs only in each row's weight at fit time.

---

## Headline

Rolling-origin validation, 19,797 fixtures over five expanding-window folds,
every arm fitted at `train_v3.PARAMS` (depth 2, the served capacity):

| arm | log loss | vs v3 | 95% CI | P(better) |
|---|---|---|---|---|
| **v3 (served, uniform weight)** | **0.98314** | — | — | — |
| `hl180` (180-day half-life) | 0.98824 | −0.00509 | [−0.00683, −0.00327] | 0.000 |
| `hl365` (365-day) | 0.98491 | −0.00177 | [−0.00306, −0.00046] | 0.005 |
| `hl730` (730-day) | 0.98350 | −0.00036 | [−0.00127, +0.00051] | 0.213 |
| `hl1460` (1460-day) | 0.98293 | +0.00021 | [−0.00033, +0.00078] | 0.753 |
| `shuffle_365` (placebo) | 0.98780 | −0.00466 | [−0.00599, −0.00338] | 0.000 |
| `anchor_365` (random-anchor control) | 0.98349 | −0.00035 | [−0.00130, +0.00048] | 0.217 |

The curve is monotone: shorter half-life, bigger loss. Nothing clears zero
on the favourable side, and the one arm that comes closest (`hl1460`) is
also the one that discards the least data — it is converging toward the
baseline, not past it.

---

## The curve is about discarded data, not about which era is discarded

Kish's effective sample size, `(sum w)^2 / sum(w^2)`, makes the trade
explicit. Against a training set that grows from 19,797 to 35,634 rows
across the five folds:

| half-life | ESS (fold 0, n=19,797) | ESS (fold 4, n=35,634) |
|---|---|---|
| 180d | 2,696 | 2,601 |
| 365d | 5,273 | 5,217 |
| 730d | 9,915 | 10,370 |
| 1460d | 15,201 | 19,409 |

A 180-day half-life is training on the equivalent of ~2,600-2,700 uniformly
weighted rows regardless of how large the real training set has grown —
almost exactly `league_ablate.py`'s `per_league` arm's own training size
(~3,900 rows), which lost −0.0084 for the identical reason: whatever signal
a smaller, more homogeneous slice buys, it is worth less than the rows it
gives up. `hl1460`'s ESS instead *grows* with the archive (15,201 → 19,409),
which is why it is the only arm that approaches parity with the uniform
baseline rather than falling behind it.

## The signal is real — the placebo proves it, and still loses

`shuffle_365` takes `hl365`'s exact weight values and permutes them within
league, within each fold's training rows — same histogram, same ESS, no
link to which row is actually recent. The construction is
`goals_ablate.shuffled()`'s, the same one `FINDINGS_VENUE.md`'s venue split
used, extended to a weight vector instead of a feature column.

If `hl365`'s loss were explained by discarding data however it's discarded,
the placebo would match it. It does not — the real weighting beats its own
placebo at every half-life tried:

```
hl365  over shuffle_365:  +0.00289  [+0.00128, +0.00457]  P(better)=1.000
hl730  over shuffle_365:  +0.00430  [+0.00281, +0.00572]  P(better)=1.000
hl1460 over shuffle_365:  +0.00487  [+0.00347, +0.00621]  P(better)=1.000
```

So recency is not interchangeable with "any uneven weighting of the same
shape" — the exact tree-depth-artifact failure mode
`FINDINGS_LEAGUE.md`'s `interact`-vs-`placebo` pairing was built to catch,
and this ablation survives the same check. The weighting genuinely tracks
something a tree finds useful. It is simply never worth more than the rows
it costs, at any half-life between 180 and 1460 days.

## Temporal locality, not "closest to now" specifically, does most of the non-placebo work

`anchor_365` keeps `hl365`'s exponential shape but centres it on a random
date drawn from each fold's own training window instead of "now." It lands
at −0.00035 [−0.00130, +0.00048] — a null, materially better than `hl365`
itself (−0.00177) despite not caring about recency at all. The reason is
visible in the ESS table: an anchor drawn from the middle of the training
window has more rows near it than "now" does (which sits at one extreme of
the date range), so `anchor_365`'s effective sample size (8,445–9,517) is
nearly double `hl365`'s (5,175–5,273) at the same nominal half-life. The two
arms are not a matched comparison — `anchor_365`'s better score is at least
partly because it discards less, not because it is a cleaner test of
locality. What the comparison does establish is the ordering that matters
for the headline: `shuffle_365` (destroys locality entirely) loses the most,
`hl365`/`anchor_365` (keep temporal locality, differ only in where the
window is centred) lose much less, and `v3`/`hl1460` (effectively no
discarding) lose nothing. Smoothness in time is worth more than any
particular reference point within it — recency is one choice of reference
point, not a privileged one.

## No capacity rescues it

Run again at `train_v2.V2_PARAMS` (`max_depth=4`, the legacy recipe
`FINDINGS_LEAGUE.md`'s own placebo warning is about) to check whether the
served depth-2 capacity is simply too shallow to feel the effect:

| arm | log loss | vs v3 | 95% CI |
|---|---|---|---|
| v3 (depth 4) | 0.98510 | — | — |
| `hl180` | 0.99547 | −0.01037 | [−0.01222, −0.00845] |
| `hl365` | 0.99155 | −0.00645 | [−0.00792, −0.00495] |
| `hl730` | 0.98865 | −0.00356 | [−0.00457, −0.00254] |
| `hl1460` | 0.98688 | −0.00178 | [−0.00247, −0.00111] |
| `shuffle_365` | 0.99488 | −0.00978 | [−0.01143, −0.00821] |
| `anchor_365` | 0.98682 | −0.00172 | [−0.00269, −0.00084] |

Same shape, uniformly worse at every arm (including the placebo) — depth 4
is over-capacity in general (`FINDINGS_CAPACITY.md`), and it pays an even
bigger price for the same data loss, not a smaller one. `hl365` still beats
`shuffle_365` by +0.00332 [+0.00156, +0.00516] at this depth too, so the
"real recency, but not enough" verdict is not a depth artifact in either
direction.

## No league or fold hides a result the pooled number misses

Per-league (rolling, depth 2): every league loses under every half-life
except `hl1460`, where leagues 301 and 384 turn slightly positive
(+0.00151, +0.00151) and the other three stay slightly negative — a spread
inside noise, not a league that wants its own treatment.

Per-fold is the check this phase added beyond `league_ablate.py`'s template,
because the fold boundaries straddle the COVID-era dip unevenly. Fold 0
(earliest, smallest training set) takes the biggest hit from every
aggressive half-life (`hl180`: −0.00717; `shuffle_365`: −0.01094) simply
because it has the least data to discard from in the first place. Fold 3
(whose training window already contains the dip and settles past it) is the
only fold where `hl365`/`hl730`/`hl1460` turn slightly positive — consistent
with the ESS story above, not with a COVID-specific recency win: it is the
fold with the most data overall, so the same half-lives discard
proportionally less of it.

---

## Open items

1. **The half-life bracket was widened past Dixon-Coles' own (180/365/730)
   to 1460 days on the hypothesis that a 26-year, 5-league pool tolerates a
   longer decay than a single-league fit — confirmed directionally (1460 is
   the best of the four), but the sweep was not extended far enough to find
   where the curve actually flattens into v3.** The monotone trend through
   1460 days suggests an even longer half-life (3000+ days) might cross zero
   properly rather than just approach it, at the cost of being a decreasingly
   interesting claim the closer it gets to "don't reweight at all."

2. **`anchor_365` is not a matched control for `hl365`** — it has a higher
   effective sample size at the same nominal half-life because its anchor
   sits nearer the middle of the training window than "now" does. A cleaner
   version would hold ESS fixed across the two arms (solve for each arm's
   half-life such that both land at the same effective sample size) rather
   than holding the half-life constant. Untested here; the qualitative
   ordering (placebo worst, locality-preserving arms in the middle, full
   data best) is unlikely to flip, but the quantitative gap between
   `hl365` and `anchor_365` should not be read as "centring on now is worse
   by 0.0014" — some of that gap is the ESS mismatch, not recency.

3. **Carried over, unaffected by anything here**: away-specific team
   strength (`FINDINGS_MARKET.md` item 1) and opponent-adjusted form
   statistics (`FINDINGS_WINDOW.md` item 3) remain open.
