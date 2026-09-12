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
   2016 cutoff, halves the training set, and buys nothing. Either drop it, or
   find an aggregation that beats `top3_rating` — which is the only performance
   feature that ranks anywhere in the importance list.
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
