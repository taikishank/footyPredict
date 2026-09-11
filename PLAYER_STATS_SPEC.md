# Spec: Player-Stats-Aware Match Outcome Model

## 1. Problem Statement

The current model (`ml/feature_lib.py`, `ml/build_features.py`, `inference-py/app/features.py`)
predicts match outcomes purely from **team-level form**: a rolling window
(`MIN_WINDOW`/`MAX_WINDOW` = 3–8 matches) of each team's own box-score stats
(goals, shots, passes, corners, possession, cards, etc. — see `STAT_IDS` in
`feature_lib.py`), averaged and fed to an XGBoost classifier.

This is blind to squad composition. A team missing its top scorer, first-choice
keeper, or entire back line to injury/suspension/rotation produces the
*exact same feature vector* as when fully fit, because `team_form()` only
ever sees aggregated team totals from past fixtures — never who actually
took the pitch. Two matches with identical recent form but wildly different
XIs are indistinguishable to the model today. This is the gap this spec
closes: incorporate **which players are actually playing** and their
individual recent performance into the feature set.

### Relationship to Phase 5 (existing spec)

`PROJECT_SPEC.md` Phase 5 already plans a Claude-web-search layer that
surfaces *qualitative* injury/lineup news alongside the model's output. This
spec is the *quantitative* counterpart: structured player data feeding the
model's own probability, not narrative context sitting next to it. The two
are complementary and should stay separate — Phase 5's Claude summary
remains useful even after this ships (it can explain *why* a squad-strength
feature moved), and this pipeline shouldn't try to re-scrape injury news
that Phase 5 already covers qualitatively.

---

## 2. Data Availability (SportMonks v3)

The ingestion layer already talks to SportMonks v3 Football
(`ingestor-go/internal/sportmonks/client.go`, `ml/fetch_all_fixtures_data.py`),
currently requesting only `participants;scores;statistics;state`. SportMonks
exposes the additional data this spec needs via:

- **`lineups.player;lineups.details`** on the fixture endpoint — starting XI,
  bench, formation position, and per-player match stats (rating, minutes,
  goals, assists, cards, shots, etc.) once a lineup is confirmed.
- **`sidelined`** — team-level injury/suspension list, available further
  ahead of kickoff than confirmed lineups but noisier/less complete.
- **`/squads/teams/{team_id}`** — current full roster, useful as the
  reference set for "who's normally available."

**Constraint that shapes this whole design:** confirmed lineups are
typically only published **~60–90 minutes before kickoff**, not days out.
The existing Upcoming tab (Phase 4) predicts fixtures up to ~3 days ahead —
for most of that window, no confirmed lineup exists yet. Any design here
must handle "predict now, lineup unknown" as the common case, not the
exception.

**Open question — must be verified before implementation starts:** whether
the current SportMonks subscription tier includes the `lineups` and
`sidelined` includes at all (they're commonly gated to higher plans). This
should be checked first, the same way the original spec flags "any step
that requires an external API key or paid tier."

---

## 3. Data Model Changes (Postgres)

Following the existing pattern in `ingestor-go/internal/store/postgres.go`
(parsed columns + a `raw JSONB` escape hatch, `CREATE TABLE IF NOT EXISTS` /
`ALTER TABLE ... ADD COLUMN IF NOT EXISTS` migrations appended to the same
schema string):

```sql
CREATE TABLE IF NOT EXISTS players (
    player_id    BIGINT PRIMARY KEY,
    name         TEXT NOT NULL,
    position     TEXT,          -- Goalkeeper/Defender/Midfielder/Attacker
    date_of_birth DATE,
    raw          JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS fixture_lineups (
    fixture_id   BIGINT NOT NULL REFERENCES fixtures(fixture_id),
    team_id      BIGINT NOT NULL,
    player_id    BIGINT NOT NULL REFERENCES players(player_id),
    is_starting  BOOLEAN NOT NULL,
    position     TEXT,
    jersey_number INT,
    raw          JSONB NOT NULL,
    PRIMARY KEY (fixture_id, player_id)
);

CREATE TABLE IF NOT EXISTS player_fixture_stats (
    fixture_id   BIGINT NOT NULL REFERENCES fixtures(fixture_id),
    player_id    BIGINT NOT NULL REFERENCES players(player_id),
    minutes_played INT,
    rating       NUMERIC,
    goals        INT,
    assists      INT,
    shots_total  INT,
    shots_on_target INT,
    key_passes   INT,
    tackles      INT,
    yellow_cards INT,
    red_cards    INT,
    raw          JSONB NOT NULL,
    PRIMARY KEY (fixture_id, player_id)
);

CREATE TABLE IF NOT EXISTS player_sidelined (
    player_id    BIGINT NOT NULL REFERENCES players(player_id),
    team_id      BIGINT NOT NULL,
    category     TEXT,      -- injury/suspension/international duty
    start_date   DATE,
    end_date     DATE,
    raw          JSONB NOT NULL
);
```

`player_fixture_stats` is intentionally the player-level mirror of the
existing `extract_team_stats()`/`STAT_IDS` mapping — same idea (SportMonks
`type_id` → named column), one grain down.

---

## 4. Ingestion Pipeline Changes

- **New Go package `internal/lineups`** (parse + store), mirroring the shape
  of `internal/odds` and `internal/upcoming`: parses SportMonks lineup/stat
  payloads into `Store.UpsertLineups` / `Store.UpsertPlayerStats` calls.
- **Historical backfill:** extend the offline fetchers
  (`ml/fetch_all_fixtures_data.py` / a new `ml/fetch_lineups_data.py`) to
  pull `lineups.player;lineups.details` for already-cached fixture IDs,
  reusing the existing rate-limit-aware retry/backoff logic
  (`compute_sleep_seconds`, `RESERVED_FOR_PROD`) rather than duplicating it.
- **New scheduled "lineup poller"** distinct from the existing Upcoming
  pipeline's 3-day window: polls confirmed lineups only for fixtures inside
  a tight pre-kickoff horizon (e.g. next ~2 hours), since that's the window
  in which SportMonks actually has lineup data to return. On finding a
  confirmed lineup it writes `fixture_lineups` and triggers a feature/edge
  recompute for that fixture so the surfaced prediction updates once real
  team news lands, instead of only at the original prediction time.
- **Cost note:** pulling lineups doesn't add extra requests for the batch
  (already-finished) pipeline — it's an `include` addition to the same
  per-fixture call. The new poller for *upcoming* fixtures is genuinely
  extra request volume against the shared hourly quota
  (`rateLimitFloor = 500` in `client.go`), so its polling interval needs to
  be sized against that budget, not just against how fresh we'd like the
  data.

---

## 5. Feature Engineering (`feature_lib.py` extension)

- **`player_form(history)`** — analogous to today's `team_form()`, but
  windowed by *appearances* rather than *fixtures* (a rotated/injured
  player accumulates history much slower than a team does). Rolling average
  over rating, minutes, goals, assists, key passes, shots, defensive
  actions, cards.
- **`squad_strength(lineup, player_forms)`** — aggregates a starting XI's
  individual `player_form` vectors into a fixed-size, position-aware team
  feature block: e.g. mean rating/attacking output by position group
  (GK/DEF/MID/FWD), a **continuity index** (what fraction of the team's
  most-used recent XI is retained this week), and a **key-player-absence
  count** (how many of the squad's top-N players by minutes/rating over the
  season are missing from this XI). This block is what actually captures
  "who's playing," not raw stat sums.
- **`predicted_lineup(team_id, before)`** — the fallback for the common case
  where no confirmed lineup exists yet (see §2's timing constraint): builds
  a "most likely XI" from each player's recent start-rate. Predictions built
  this way should carry a `lineup_source` (`confirmed` vs `predicted`) flag
  through to the API response, since a predicted-lineup-based prediction is
  materially less reliable than one built off a confirmed team sheet — the
  frontend/edge logic downstream should be able to treat these differently
  rather than presenting both with equal confidence.
- **Missing-history imputation:** unlike teams (which have a hard
  `NotEnoughHistoryError` gate at `MIN_WINDOW`), a lineup will very often
  contain at least one debutant/new signing with no `player_form` history.
  Erroring out on that basis would make the model unusable most weeks, so
  this needs an explicit fallback (e.g. league-and-position average) rather
  than inheriting the team-level all-or-nothing contract.

---

## 6. Model Changes

- Extend `build_match_features()` / `build_feature_table()` to append the
  new `squad_strength` block to the existing team-form vector, with an
  explicit `PLAYER_FIELD_ORDER` constant (same role as today's
  `FORM_FIELD_ORDER`) so training and serving stay in lockstep the same way
  they do today.
- Train a **challenger** model against this expanded feature set, logged to
  its own MLflow experiment (e.g. `liveedge-match-outcome-playerstats`)
  rather than overwriting the current run history — keeps the incumbent
  directly comparable.
- **Promotion bar:** the challenger must beat the incumbent's log loss/Brier
  score on the same chronological validation split by more than noise (e.g.
  a paired bootstrap over the validation set), not just "adding more
  features changed the number." Added complexity that doesn't clearly help
  isn't worth shipping.

---

## 7. Serving & Rollout Plan

- Tag predictions with a `model_version` distinguishing e.g. `team-form-v1`
  vs `team-player-v2` (extends the existing `model_version()` scheme in
  `inference-py/app/model.py`), so `edges`/`prediction_calls` can be sliced
  by model version in `backtest/report.py` / `grade_calls.py`.
- **Shadow mode first:** run the challenger alongside the incumbent on every
  `/fixtures/upcoming` call for a period, logging its probabilities without
  letting it drive the surfaced edge/flag. This validates the challenger
  against live, not just historical, data before it affects anything a user
  sees.
- **Cutover** only once the challenger beats the incumbent on live-graded
  `prediction_calls` (not just the offline backtest) over a meaningful
  sample.
- **Rollback:** keep both model artifacts in S3 under distinct keys (mirrors
  the existing `MODEL_S3_KEY`/`S3_BUCKET_MODELS` config pattern) so
  reverting is a config change, not a re-train.

---

## 8. Non-Goals (this iteration)

- No player tracking / xG-model-grade data — would require a different data
  provider than SportMonks' box-score-level stats.
- No live/in-match integration — this scopes to pre-match (Upcoming tab)
  predictions only. Feeding confirmed-XI or in-match substitution data into
  the live win-probability layer (Phase 3) is a future phase, not this one.
- Not a replacement for Phase 5's Claude qualitative injury/news synthesis —
  this is the structured, quantitative complement to it.

---

## 9. Open Questions / Risks

1. Does the current SportMonks plan actually include `lineups` and
   `sidelined`? Confirm before committing engineering time.
2. What's the real request/payload cost of backfilling lineups across the
   ~2000+ already-cached historical fixtures, against the shared hourly
   quota noted in `ml/fetch_all_fixtures_data.py`?
3. How far ahead of kickoff are lineups reliably confirmed for the leagues
   this model actually covers — does it vary enough by competition to need
   per-league handling?
4. How much of the Upcoming tab's 3-day window will realistically still be
   served by `predicted_lineup()` rather than a confirmed one, and is that
   still enough of an improvement over team-form-only to be worth the
   added pipeline complexity?

---

## 10. Suggested Phases (continuing `PROJECT_SPEC.md`'s numbering)

- **Phase 8 — Data model & historical backfill.** Confirm API coverage
  (§9.1), add the new tables (§3), backfill lineup/player-stat history for
  cached fixtures.
- **Phase 9 — Player-form feature library & offline comparison.** Build
  `player_form`/`squad_strength` in `feature_lib.py`, train the challenger,
  compare against the incumbent per §6's promotion bar.
- **Phase 10 — Live lineup ingestion.** Build the lineup poller and
  `predicted_lineup()` fallback (§4–5).
- **Phase 11 — Serving integration & shadow mode.** Wire the challenger into
  `inference-py`, add `model_version`/`lineup_source` tagging, run shadow
  mode (§7).
- **Phase 12 — Cutover & dashboard surfacing.** Promote the challenger once
  it clears the live bar; surface `lineup_source`/confidence on the
  frontend so users can see when a prediction is based on a confirmed vs.
  predicted XI.
