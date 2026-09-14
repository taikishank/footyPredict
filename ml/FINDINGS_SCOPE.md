# Train/serve skew: which form window, and what did the mismatch cost?

`FINDINGS_RATINGS.md` open item 4 and `FINDINGS_VENUE.md` open item 5 both
name the same defect and neither measured it. This phase measures it and
closes it.

`ml/build_features.py:96` keeps one unbounded list of prior matches per club
and averages the last `MAX_WINDOW = 8`, whatever season they fall in.
`inference-py/app/db.py:76` scoped the same query to the fixture's own
`season_id`. Both have been true since v2 shipped, so the served model was
fitted under one recipe and queried under another.

New: `ml/scope_lib.py` (the two recipes), `ml/build_scope_features.py` (one
replay emitting both), `ml/scope_ablate.py`, `ml/test_scope_lib.py`.
`ml/build_features.py` gained `season_id` on its parsed fixtures and nothing
else - the feature table it writes is byte-identical, asserted below.

---

## How far apart the two recipes were

Replaying the archive (40,098 played fixtures):

| | |
|---|---|
| Training rows carrying ≥1 prior-season match in their 16 window slots | 20.4% |
| Training rows that could never be served (neither club has `MIN_WINDOW` in-season) | 7.4% (2,936) |
| August fixtures servable | **11.6%** (September 82.4%, October 99.2%) |
| Training rows averaging a *partial* window (n < 8) | 2.0% |
| Fixtures served from a partial window (n = 3–7, in-season only) | 13.7% |

Two defects, not one. **Coverage**: 88% of August had no served prediction
while the offline table scored those fixtures happily. **Distribution**:
`team_form()` divides by `len(window)`, so the service emitted 3-to-7-match
means on ~every early-season fixture, a shape the model saw on 2% of its
training rows. The flattering offline August number (0.9917 against a 0.993
mean) was computed from eight matches of carried-over form the service did
not have.

---

## Arms

| arm | window | gate | |
|---|---|---|---|
| `xseason` | cross-season | ≥3 | the training recipe; the baseline every published number was measured under |
| `xseason_age` | cross-season | ≥3 | + `matches_into_season`/`stale_slots` per side — the cheap fix: tell the model its window is stale, change no query |
| `scoped` | in-season only | ≥3 in-season | the serving recipe, exactly as `db.py` ran it |
| `scoped_full8` | in-season only | ≥8 in-season | no partial windows ever; refuses early-season fixtures outright |

Two tables, because the arms deliberately cover different fixtures and each
table is dishonest in the way the other is not. **PAIRED** runs on the 15,454
fixtures every arm emits — a valid paired bootstrap, and the table that
flatters `scoped_full8`, since the rows it refuses are exactly the rows
excluded. **PRICED** runs on all 19,810 and scores an arm's refusals at the
fold's class base rate, because refusing is a real product outcome (the
Upcoming tab falls back to an odds-only row; `/predict` returns 422).

Machinery is `rating_ablate.py`'s, imported: same expanding-window folds, same
`train_v2` hyperparameters, same paired bootstrap. `--mode rolling` below;
`--mode split` agrees on every sign.

---

## Result: the scoping bought nothing and cost 8.3% of the fixtures

**PAIRED** (15,454 fixtures)

| arm | log loss | vs `xseason` | 95% CI | P(better) |
|---|---|---|---|---|
| `xseason` | 0.99188 | — | | |
| `xseason_age` | 0.99147 | +0.00041 | [+0.00009, +0.00076] | 0.993 |
| `scoped` | 0.99189 | −0.00001 | [−0.00058, +0.00057] | 0.485 |
| `scoped_full8` | 0.99341 | −0.00153 | [−0.00218, −0.00089] | 0.000 |

**PRICED** (19,810 fixtures, refusals at the base rate)

| arm | log loss | vs `xseason` | 95% CI | P(better) |
|---|---|---|---|---|
| `xseason` | 0.99292 | — | | |
| `xseason_age` | 0.99254 | +0.00038 | [+0.00004, +0.00069] | 0.989 |
| `scoped` | 0.99999 | −0.00707 | [−0.00866, −0.00544] | 0.000 |
| `scoped_full8` | 1.01221 | −0.01930 | [−0.02169, −0.01691] | 0.000 |

**Coverage**

| arm | served | Aug | Sep |
|---|---|---|---|
| `xseason` | 100.0% | 100.0% | 100.0% |
| `scoped` | 91.7% | 8.9% | 82.3% |
| `scoped_full8` | 78.0% | 0.7% | 0.0% |

`scoped` is a dead null against `xseason` where both can be computed
(−0.00001, an interval of ±0.0006) and loses −0.00707 once its refusals are
priced. **The entire difference between the two live recipes is coverage.**
Season-scoping was not a worse feature; it was the same feature, computable
on 8.3% fewer fixtures.

---

## The partial window is not the problem either

The distribution defect on its own, with coverage held out — August and
September fixtures `scoped` *can* serve, so every cell is a real model
prediction and none is a priced refusal:

**EARLY SEASON** (1,781 fixtures)

| arm | log loss | vs `xseason` | 95% CI | P(better) |
|---|---|---|---|---|
| `xseason` | 0.99515 | — | | |
| `xseason_age` | 0.99541 | −0.00027 | [−0.00139, +0.00082] | 0.308 |
| `scoped` | 0.99436 | +0.00078 | [−0.00329, +0.00512] | 0.633 |

A three-match in-season mean predicts August as well as an eight-match
cross-season one, if anything marginally better. The model tolerating an
input shape it was barely trained on is a genuine result and not the one
expected: the skew was real, and in log-loss terms it was free.

What was *not* free is the by-month PRICED table, which is the same finding
from the other side:

| month | n | `xseason` | `scoped` | `scoped_full8` |
|---|---|---|---|---|
| Aug | 1,375 | **0.98803** | 1.07101 | 1.07894 |
| Sep | 2,016 | **0.99915** | 1.01050 | 1.07919 |
| Oct | 2,004 | 0.98753 | 0.98854 | 1.01900 |
| Nov–Jul | — | within ±0.002 of each other | | |

August is +0.083 and it is all refusals. From a club's ninth in-season match
the two recipes are arithmetically identical, so everything from October on
is fold noise — which is exactly why the full-archive mean hid this for the
life of the model.

---

## Three things worth recording

**1. `scoped_full8` loses on rows where its inputs are identical.** −0.00153,
significant, on the intersection — where it refuses nothing and reads the
same columns as `scoped`. The arms differ there only in what they *trained*
on: 31,267 rows against 36,683. Refusing to average a partial window throws
away 15% of the training set to avoid an input that the early-season table
just showed costs nothing. It is the worst arm on both tables and there is no
reading under which it wins.

**2. `xseason_age` is significant and probably not real.** +0.00041,
P = 0.993, on both tables — but −0.00027 and null in early season, which is
the only place its stated mechanism exists. `stale_slots` is 0 from the ninth
in-season match, so a column that can only vary in August and September
cannot honestly be earning its keep in a number driven by the other ten
months. The likeliest reading is that `matches_into_season` is acting as a
generic season-progress proxy and four extra columns bought a little split
capacity. Not shipped; +0.0004 is inside the noise every other phase has
called null, and the one diagnostic that could support it points the other
way.

**3. The published Aug/Sep numbers were measured on inputs the service never
produced.** `FINDINGS_RATINGS.md` recorded August and September as among the
*better*-predicted months. They are — under `xseason`, which is now what runs.
Under the recipe that was actually serving, August was 1.071.

---

## Change made

`inference-py/app/db.py` — `get_team_history()` loses its `season_id`
parameter and the `(raw ->> 'season_id')::bigint = %s` predicate;
`app/features.py` stops passing one. That is the whole fix. It aligns serving
to training rather than the reverse, needs no retrain and no migration, and
restores predictions on the ~8.3% of fixtures (91% of August) that previously
fell back to an odds-only row.

`--mode split` is listed here for comparability with every number in
`FINDINGS_PHASE9.md`: `scoped` +0.00010 [−0.00076, +0.00100] paired, −0.00810
priced. Same signs, same conclusion, one fold.

The `__xseason` columns of `build_scope_features.py` reproduce
`data/features/2000_window3_8.csv` on all 57 shared columns over 39,594
shared fixtures. That assertion is what makes the baseline arm the training
recipe rather than a lookalike, and it is the reason the −0.00001 above can
be read as "these are the same feature".

---

## Open items

1. **Cross-season form now spans divisions.** A promoted club's August window
   carries second-tier matches. Training has always done this, so the skew is
   closed either way and the ablation covers it — but nothing has asked
   whether a promoted side's carried-over form is *worth* anything, and it is
   a different question from the one measured here.

2. **Ratings are immune and were held out.** `atkdef` is explicitly
   cross-season, so no rating column could move between these arms; this
   phase ran on the served v2 recipe (form + `club_goals_relative`)
   deliberately. If a `train_v3.py` is built, the arms need no re-run — the
   scoping question is settled independently of the rating columns.

3. **The coverage recovered is not an accuracy gain.** −0.00707 is the price
   of refusals against a base-rate fallback, not model improvement, and it
   does not close any part of the 0.0211 market gap. What it does is make
   ~1,600 fixtures a year predictable that previously were not, and make
   every by-month number in this project mean what it says.

4. **`scoped_full8`'s training-set finding generalises.** Any future arm that
   tightens a gate pays for it in training rows as well as coverage, and the
   PAIRED table will not show that cost. Worth remembering the next time a
   gate looks like a free way to clean up an input distribution.

5. **Still untouched, and now the top of the list**: the Dixon–Coles goals
   model, and opponent-adjusting the 54 form columns. Both were ranked above
   this phase on expected gain, and this phase was never expected to be one.
