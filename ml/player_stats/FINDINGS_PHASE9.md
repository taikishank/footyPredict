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
```

Nothing here touches `ml/feature_lib.py`, `ml/build_features.py` or `ml/train.py`.
The incumbent's training input is byte-identical and its MLflow experiment is
untouched; challenger runs go to `liveedge-match-outcome-playerstats`.

---

## Headline

**Squad features clear the bar, but not for the reason the spec expected.**

| comparison | log loss reduction | 95% CI | verdict |
|---|---|---|---|
| team form → + availability block (2006+, 7,091 val fixtures) | **+0.00608** | [+0.00267, +0.00974] | clears |
| team form → + availability block (2016+, 3,601 val fixtures) | **+0.00690** | [+0.00277, +0.01100] | clears |
| + availability → + performance block (2016+) | +0.00049 | [−0.00412, +0.00498] | **within noise** |

The 14 minutes-derived availability features capture the entire gain. The 36
rating-derived performance features — the ones that cost the 2016 cutoff and
half the training history — add nothing measurable on top.

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
still unproven, and the reason is probably that **key-player absence is being
measured against a season-minutes ranking rather than against a player's actual
importance**. A club's 11th-most-used player missing counts the same as its top
scorer missing. Worth revisiting before Phase 10 builds serving on top of it.

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

1. **Key-player absence needs a better importance measure than season minutes**
   (see above). This is the single most likely reason §1's central claim is not
   showing up in the numbers.
2. **The performance tier should not be carried forward as-is.** It costs the
   2016 cutoff, halves the training set, and buys nothing — and the section
   above rules out both column count and missing stat types as the cause. If it
   is revisited, the thing to change is the *aggregation*, not the input:
   `top3_rating` outranks every other performance feature, which points at the
   distribution of quality across an XI rather than its per-position mean.
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
