# Phase 9 findings: the player-form feature library and the offline comparison

Answers `PLAYER_STATS_SPEC.md` §5–6: build `player_form`/`squad_strength`, train
a challenger, and judge it against §6's promotion bar — a paired bootstrap
improvement, not just "adding features changed the number".

Reproduce with:

```
./inference-py/bin/python ml/player_stats/build_player_table.py      # ~1 min
./inference-py/bin/python ml/player_stats/build_player_features.py   # ~3 min
./inference-py/bin/python ml/player_stats/train_challenger.py --tier availability
./inference-py/bin/python ml/player_stats/train_challenger.py --tier performance
./inference-py/bin/python ml/player_stats/ablate.py blocks   --tier performance
./inference-py/bin/python ml/player_stats/ablate.py features --tier availability
./inference-py/bin/python ml/player_stats/ablate.py trim     --tier performance
./inference-py/bin/python ml/player_stats/ablate.py weighting --tier availability
./inference-py/bin/python ml/player_stats/ablate.py weighting --tier performance
```

Nothing here touches `ml/feature_lib.py`, `ml/build_features.py` or `ml/train.py`.
The incumbent's training input is byte-identical and its MLflow experiment is
untouched; challenger runs go to `liveedge-match-outcome-playerstats`.

---

## Headline

**Squad features clear the bar, but not for the reason the spec expected.**

| comparison | log loss reduction | 95% CI | verdict |
|---|---|---|---|
| team form → + availability block (2006+, 7,091 val fixtures) | **+0.00618** | [+0.00276, +0.00982] | clears |
| team form → + availability block (2016+, 3,601 val fixtures) | **+0.00731** | [+0.00304, +0.01142] | clears |
| + availability → + performance block (2016+) | +0.00060 | [−0.00393, +0.00539] | **within noise** |

The 18 minutes-derived availability features capture the entire gain. The 46
rating-derived performance features — the ones that cost the 2016 cutoff and
half the training history — add nothing measurable on top.

(Both blocks grew in the weighted rework below, from 14 and 36 columns; the
earlier numbers for the narrower blocks were +0.00608, +0.00690 and +0.00049 —
the same verdicts, and the differences are an order of magnitude inside what
this split resolves.)

Reweighting the aggregations so that players are no longer interchangeable
(below) does not change that: it is a null result, and an informative one.

---

## Why the comparison is built the way it is

**The baseline had to be retrained.** The shipped model sees 39,594 fixtures
back to 2000; a rating-aware challenger can only see the ~18,000 from 2016
(Phase 8 established the 2015/16 provider cliff). Comparing the two as they
stand would have measured the lost decade, not the added features. Both arms in
`train_challenger.py` are therefore trained here, on identical rows, an
identical chronological 80/20 split, and the incumbent's own hyperparameters —
differing only in which columns they get.

**The two tiers exist so the question can be asked twice.** Minutes played is
populated from 2006 (96.8% of starters) and ratings only from 2016 (99.6%
there, 5.5% before). Splitting the block along that line lets Phase 9 ask *does
knowing who plays help* over 35,453 fixtures and *does knowing how well they
play add anything* over 18,004, instead of collapsing both into one confounded
test on the smaller set.

**The bootstrap is paired.** Both arms score the same validation fixtures, so
resampling the per-fixture loss *difference* lets a chaotic week inflate both
losses together and cancel. Unpaired, a ~0.006 effect would sit well inside the
spread.

**Training noise turned out to be exactly zero.** Five XGBoost seeds gave
bit-identical validation losses — with no `subsample` or `colsample`, the
incumbent's configuration is fully deterministic. So the bootstrap CI is the
whole error story, which is not something to assume of a tree model in general.

---

## What actually carries the signal

Ablating the availability block on the 2006+ tier:

| arm | columns | log loss | vs. form only |
|---|---|---|---|
| team form only | 54 | 0.99979 | — |
| + `xi_relative_experience` | 56 | 0.99444 | **+0.00536**, CI [+0.00197, +0.00887] |
| + the other six availability features | 66 | 0.99848 | +0.00131, CI [+0.00004, +0.00261] |
| + all | 68 | 0.99371 | +0.00608 |

**Two columns out of sixty-eight carry ~88% of the gain**, and adding the
remaining six on top of them is within noise ([−0.00026, +0.00171]).

That is worth stating plainly rather than dressing up: what the model has
learned is closer to *how experienced this starting eleven is relative to its
league* than to *team news*. Rotation, key-player absence and debutant count do
clear on their own, but barely — the lower CI bound is +0.00004 — and they are
redundant once experience is present. The spec's framing (§1: "a team missing
its top scorer produces the exact same feature vector") is only weakly
supported by this evidence; the measurable effect is a squad-quality proxy.

This does not make the block worthless — it is a real, reproducible improvement
the incumbent cannot express — but it does mean the interesting claim in §1 is
still unproven, and the suspected reason was that **key-player absence is being
measured against a season-minutes ranking rather than against a player's actual
importance**: a club's 11th-most-used player missing counts the same as its top
scorer missing.

That suspicion has since been tested and does not hold — see "The weighted
rework" below. Weighting absence by minutes is nearly collinear with counting
it, because the same minutes pick the key eleven in the first place. §1 remains
unproven, and the reason lies elsewhere.

---

## The performance block is weak, not redundant, and trimming does not fix it

The nested comparison above shows the performance block adding nothing *on top
of* availability, which has two very different explanations: the ratings could
be carrying the same information the minutes features already carry, or they
could be carrying almost nothing. Running the block on its own separates them
(`ablate.py blocks`, 2016+, 3,601 validation fixtures):

| arm | columns | log loss | vs. form only |
|---|---|---|---|
| form only | 54 | 1.00334 | — |
| form + availability | 68 | 0.99643 | **+0.00690**, clears |
| form + performance | 90 | 1.00221 | +0.00113, noise |
| form + both | 104 | 0.99594 | +0.00740, clears |

**Weak, not redundant.** Without the availability block to lean on, 36 rating
features buy +0.00113 over bare team form — six times less than 14 minutes
features buy, and well inside noise. Adding availability *to* performance still
clears (+0.00627), so the direction of redundancy is one-way.

### Was it simply too many columns for 14,403 training rows?

A fair worry, and the reason `ablate.py trim` exists: rank the performance
features by importance from the train-fit model, keep the top k home/away
pairs, and re-test.

| top-k | columns | log loss | vs. form + availability |
|---|---|---|---|
| 1 (`top3_rating`) | 70 | 0.99487 | +0.00157, CI [−0.00093, +0.00408] |
| 4 | 76 | 0.99554 | +0.00089, CI [−0.00231, +0.00412] |
| 8 | 84 | 0.99398 | +0.00246, CI [−0.00136, +0.00614] |
| 18 (all) | 104 | 0.99578 | +0.00065, CI [−0.00398, +0.00513] |

**No k clears.** The best raw log loss (k = 8) is also the best of eight tries
on one validation set, which makes it the least trustworthy number in the
table, and its interval still spans zero. Column count is not the lever in
either direction.

### Stage 1 did not capture everything, and that is not the problem either

For the record, since the opposite question is the natural one to ask: stage 1
captures **18 of the 74 stat types** present in the dense era. Widely-available
types left out include Passes, Accurate Passes, Total/Accurate Crosses, Total
Duels, Aerials Won, Fouls, Fouls Drawn, Dispossessed, Offsides, Big Chances
Missed and Captain — all on ~85-91% of 2016+ fixtures. Adding them is one
re-run of stage 1 away, because Phase 8 deliberately cached every type.

But the evidence above says more raw stat types is the wrong direction. What is
failing is not the breadth of the input, it is that a per-position mean of
rolling per-90 rates does not distinguish two XIs the way team form already
does. `top3_rating` ranks first among the performance features and is the only
one that ranks near the availability features at all, which hints the useful
signal is in the *distribution* of quality across an XI rather than its
average.

### What this split can and cannot see

Every "noise" verdict here has a floor: the 2016+ validation set is 3,601
fixtures and the paired bootstrap's interval half-width is ~0.0045. So the
honest statement is not "the performance block does nothing" but **"any effect
it has is smaller than ~0.005 log loss"** — an order of magnitude below the
+0.0069 the availability block delivers on the same rows. Distinguishing a
genuinely-zero effect from a small one would need either more history (which
the 2015/16 provider cliff forecloses) or a much larger validation set.

---

## The weighted rework: a null result, and why

Open items 1 and 2 below both blamed the *aggregation* rather than the input:
`key_players_absent` counts heads, so a club's 11th-most-used player missing
scores the same as its top scorer, and the only performance feature that ranked
at all was `top3_rating`, a distribution statistic. The rework weights both by
how much each player actually plays:

| field | tier | what it replaces |
|---|---|---|
| `key_minutes_absent` | availability | the count, by share of the key eleven's season minutes |
| `top_player_absent` | availability | — (the sharpest form of §1: the ever-present man is out) |
| `xi_weighted_rating` | performance | the flat per-position mean, weighted by season minutes |
| `rating_top1`, `rating_bottom3` | performance | — (the distribution `top3_rating` half-described) |
| `absent_rating_mean`, `absent_rating_weighted` | performance | — (the absentees carry their own rolling form) |

`ablate.py weighting` swaps rather than stacks the two absence forms, because
the claim is that the weighted column says something the count cannot; helping
only *alongside* the count would be a much weaker result.

Availability tier (2006+, 7,091 validation fixtures):

| arm | columns | log loss | vs. form only |
|---|---|---|---|
| form only | 54 | 0.99979 | — |
| counted absence | 68 | 0.99371 | +0.00608, clears |
| weighted absence | 70 | 0.99368 | +0.00611, clears |
| both forms | 72 | 0.99362 | +0.00617, clears |

Performance tier (2016+, 3,601 validation fixtures):

| arm | columns | log loss | vs. previous |
|---|---|---|---|
| counted absence | 68 | 0.99643 | — |
| weighted absence | 70 | 0.99582 | +0.00062 on the swap |
| + flat performance | 108 | 0.99519 | +0.00084, noise |
| + weighted squad quality | 114 | 0.99514 | +0.00005, noise |
| + absent quality | 112 | 0.99544 | −0.00025, noise |
| + all weighted | 118 | 0.99543 | −0.00024, noise |

**Every weighted variant lands inside noise.** The best of them — swapping the
weighted absence in for the count on 2016+ — is +0.00062, an eighth of what this
split can resolve. Nothing here is worse either; the two parameterisations are
empirically the same feature.

### Why, which is the part worth keeping

**Weighting absence by minutes is close to circular.** `key_minutes_absent`
correlates **+0.981** with the count it was meant to replace. Conditioning on
the count shows why:

| key players absent | mean minutes share absent | std | p10–p90 |
|---|---|---|---|
| 1 | 0.076 | 0.016 | 0.054–0.097 |
| 2 | 0.158 | 0.022 | 0.130–0.188 |
| 3 | 0.244 | 0.027 | 0.210–0.278 |
| 5 | 0.424 | 0.033 | 0.381–0.466 |

Given *k* absences the weighted feature is *k*/11 ± 0.03. Of its 0.139 total
standard deviation, only **0.027 survives** removing the count. The reason is
structural: the key eleven are *selected* by season minutes, and selecting the
eleven most-used players selects eleven players with similar minutes. Ranking by
minutes and then weighting by minutes asks the same question twice. The gap
between a club's most- and 11th-most-used player is about four percentage points
of first-team playing time, which is not enough spread for a tree to split on.

**And the missing players are not better than the ones replacing them.** Across
18,004 fixtures the absent key players' mean rolling rating is **6.96** against
the fielded XI's weighted **7.03**. The absences in this data are mostly
rotation, not stars ruled out — so an absence-quality feature is averaging over
a population that is, if anything, slightly below the side that takes the pitch.
That is a fact about what "key player absent" means in a season's worth of team
sheets, not a property of the formula.

**The weighted ratings barely move either.** `xi_weighted_rating` correlates
+0.922 with `top3_rating` and +0.82 with the flat positional mean it was meant
to improve on. Re-weighting eleven numbers that already sit inside a 0.15-wide
band cannot produce a materially different column.

It does, though, become the performance block's *preferred* column:
`away_sq_xi_weighted_rating` now ranks third in the whole challenger by
importance, above every team-form feature except passing, and takes the place
`top3_rating` held. That is a useful warning about importance rankings — the
model reaches for the column, and the paired bootstrap says reaching for it
changes the predictions by +0.00005. Importance says which column a tree
prefers among correlated alternatives, not how much the block is worth.

### What this settles

Open items 1 and 2 are **closed as falsified rather than fixed**. The
aggregation was a reasonable suspect and it was not the lever: §1's claim does
not fail because absence is counted instead of weighted, and the performance
block is not weak because its means are unweighted. What the evidence now says
is that **season minutes — in any parameterisation, counted or weighted — is the
wrong importance measure**, because it is the same quantity used to pick the key
players in the first place. A measure with independent variation (share of the
club's goal involvements, or minutes *missed* against a known injury list rather
than inferred from the team sheet) would be a different test; this one is done.

The weighted columns stay in the library. They cost two columns in the
availability tier, are not worse on either tier, have the better point estimate
on both, carry no date correlation (+0.02 and −0.01), and are the more defensible
thing to serve. But they are kept as a no-cost improvement in parameterisation,
**not** as a measured gain — and the headline result is unchanged: the
availability tier clears the bar, the performance tier does not.

---

## Design decisions taken

**Two stages, two artifacts.** `build_player_table.py` flattens all 41,777
gzipped payloads into `player_matches.parquet` (1,416,836 player-match rows,
38,026 fixtures, 18,892 players) in ~1 minute; `build_player_features.py` reads
that parquet in seconds. Feature design is iterated a dozen times, and paying
the ~1-minute decompression on every iteration would have been the single
biggest tax on this phase.

**Wide at stage 1, despite `parse_lineups.py` being deliberately long.** The
long form's advantage — never migrating when a new stat type matters — is
preserved by stage 1 being cheap to re-run against an untouched cache, and
every consumer downstream wants "this player's rating in this match" as a
column.

**Starters only.** Phase 8 Q4 flagged that training is all confirmed XIs while
serving inside 48h will be all predicted ones. A predicted-XI payload is 22
starter rows with no bench, so reading only `is_starting` rows makes the
confirmed and predicted cases produce *structurally identical* vectors. The
train/serve gap is thereby reduced to how often the provider guesses the XI
wrong, which is a Phase 10 measurement, not a feature-shape problem. Adding
bench-depth features would reintroduce it.

**Per-90 rates over window totals, not per-match means.** A five-minute cameo
with a goal reads as one goal inside a ~900-minute window, not as eighteen
goals per 90.

**Position-group means plus counts.** Eleven variable-length player vectors
have to collapse to a fixed-size block. Grouping by GK/DEF/MID/FWD gives the
position counts for free, which encode the shape of the XI — a back three
versus a back four — without anyone parsing a formation string.

**Imputation is league-and-position average plus a `imputed_players` count**
(spec §5, plus the promoted-team case Phase 8 flagged). `PositionAverages` is a
stateful object walked in date order and updated strictly after each fixture's
features are emitted, so a player's own match can never contribute to the
average used to impute him. A groupby over the whole table would have been one
line and would have leaked the future into every row.

---

## Two bugs worth recording

**`NaN or 0` is `NaN`.** The parquet carries unrecorded ratings and minutes as
NaN, NaN is truthy in Python, and the first build therefore produced feature
blocks that were 97–99% null — which reads on inspection as "the archive is
thin here" rather than "the code is wrong". Fixed with an explicit `numeric()`
coercion used at every read, and pinned by a test. Missing minutes are now
treated as *not an appearance* rather than as zero minutes: a bench player who
never came on and one whose minutes were never reported are indistinguishable
in this data, and counting either would dilute every form window.

**Career appearance count was an era proxy.** The first version of the
experience feature was the XI's mean career appearances. Our archive starts in
2006, so a 2006 starter with ten seasons behind him reads as a debutant, and
the league-wide mean climbs from 22.9 in 2006 to ~120 by 2018 purely as the
observation window fills up. That ramp correlates **+0.50 with the fixture
date**, and it was the second-strongest feature in the model — on a
chronological split, a model can score well on it for entirely the wrong
reason. Replaced by `xi_relative_experience`: the same quantity divided by a
rolling 200-fixture league mean, which is what "experienced" was supposed to
mean. Date correlation drops to −0.06, the yearly mean flattens to ~1.0, and
the measured improvement *survives* — 0.99444 against 0.99979 — so the signal
was real and only its parameterisation was leaking.

---

## Open items for Phase 10

1. ~~**Key-player absence needs a better importance measure than season
   minutes.**~~ **Tested and closed** — see "The weighted rework" above. A
   minutes-weighted absence feature correlates +0.981 with the count it
   replaced and changes nothing, because the key eleven are selected by minutes
   in the first place. The open item that survives is narrower and harder: an
   importance measure with *independent* variation. Goal involvements are in
   the cache already; a real injury list is not, and "absent from the team
   sheet" conflates injury with rotation, which the 6.96-vs-7.03 rating
   comparison above shows is the dominant case.
2. **The performance tier should not be carried forward as-is.** It costs the
   2016 cutoff, halves the training set, and buys nothing. The three
   explanations worth ruling out now have been: column count (`trim`, no k
   clears), breadth of input (18 of 74 stat types captured, and more would not
   help), and the aggregation (`weighting`, every variant inside noise). The
   flat per-position mean was the best remaining suspect and it is not the
   cause — re-weighting an XI whose ratings all sit inside a 0.15-wide band
   cannot produce a different column. Whatever is missing is not in this
   feature family.
3. **The train/serve gap is still unmeasured.** The cache holds only ~40
   fixtures with a predicted XI (874 rows), far too few to estimate how much
   accuracy is lost when the XI is guessed rather than confirmed. Phase 10's
   poller should accumulate predicted XIs alongside the confirmed ones that
   follow, which turns this into a directly measurable quantity.
4. **`days_since_last_match` reaches 5,942 days** for a club returning from a
   long absence from the top five leagues. Trees handle it, but it is worth
   knowing the column is not the winter-break indicator it looks like.
5. **Positions are missing on ~15% of 2006–2015 starters** and ~0.03% of 2016+
   ones. The availability tier is position-agnostic so this costs nothing today;
   it would matter if position-aware features were extended back before 2016.
