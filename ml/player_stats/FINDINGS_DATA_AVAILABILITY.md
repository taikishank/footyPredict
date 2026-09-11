# Phase 8, step 0 findings: SportMonks player-data availability

Answers `PLAYER_STATS_SPEC.md` §9 Q1–Q4, which the spec says to settle before
committing engineering time. Reproduce with:

```
./inference-py/bin/python ml/player_stats/probe_availability.py
```

Raw payloads land in `ml/data/raw/player_stats_probe/` (gitignored-sized, not
committed) so the rest of Phase 8 can be built offline against real shapes.

Probed 2026-09-11 against plan **"Taiki Shank Custom Plan 1"** (5 leagues:
Premier League, Bundesliga, Ligue 1, Serie A, La Liga).

---

## Q1 — Does the plan include `lineups` and `sidelined`? **Yes, mostly.**

`GET /v3/my/enrichments` (38 granted) includes:

| enrichment | relevance |
|---|---|
| Access lineups | the core requirement — §3's `fixture_lineups` + `player_fixture_stats` |
| Access players | §3's `players` table |
| Access Predicted Lineups | **changes the §5 design, see Q3** |
| Access Formations | formation position, already on each lineup row |
| Access statistics | what we already use |

Not granted: anything matching *sidelined*, *squad*, or *injury*.

Empirically, on the most recent finished fixture in each of the five leagues,
every include returns real data — including `sidelined`:

| include | result |
|---|---|
| `lineups` | OK — 39–49 rows per fixture (XI + bench, both teams) |
| `lineups.player` | OK |
| `lineups.details` | OK — per-player stats attached |
| `sidelined` | OK — 1–7 rows per fixture |
| `statistics` | OK (control) |

So `sidelined` **is** available as a fixture include despite not appearing in
the enrichment list. What is *not* available:

- `GET /squads/teams/{id}` → **HTTP 403** (gated). §2 wanted this as the
  "who's normally available" reference set; it has to be derived from
  appearance history instead (which `player_fixture_stats` gives us anyway).
- `GET /sidelined` as a standalone endpoint → **HTTP 404**; it exists only as
  a per-fixture include in v3, so there's no cheap "all current injuries"
  sweep. `/my/resources` lists no squad or sidelined endpoint either.

**Verdict: §3's four tables are all populatable**, except that
`player_sidelined` must be filled from the per-fixture `sidelined` include
rather than a squad/injury endpoint.

### ⚠️ Unrelated but urgent: the subscription expires 2026-09-12 22:24 UTC

Every response's `subscription.meta` reports `ends_at: 2026-09-12 22:24:36` —
**tomorrow**. Phase 8 is moot if that lapses, and so is the existing pipeline.
Worth checking before anything else here.

### Per-player stat types confirmed (§3's `player_fixture_stats`)

60 distinct `type_id`s come back on `lineups.details`. Every column §3 names
exists:

| §3 column | SportMonks type_id |
|---|---|
| `minutes_played` | 119 (Minutes Played) |
| `rating` | 118 (Rating) |
| `goals` | 52 (Goals) |
| `assists` | 79 (Assists) |
| `shots_total` | 42 (Shots Total) |
| `shots_on_target` | 86 (Shots On Target) |
| `key_passes` | 117 (Key Passes) |
| `tackles` | 78 (Tackles) |
| `yellow_cards` | 84 (Yellowcards) |
| `red_cards` | 83 (Redcards) — type exists; absent from the probed fixture only because no red card was shown |

Useful extras beyond the spec, for `squad_strength` (§5): 100 Interceptions,
101 Clearances, 106 Duels Won, 109 Successful Dribbles, 9706 Chances Created,
580 Big Chances Created, 88 Goals Conceded, 57 Saves, 40 Captain.
Full map cached at `ml/data/raw/player_stats_probe/core_types.json`.

Each lineup row also carries `team_id`, `player_id`, `player_name`,
`position_id`, `formation_field`, `formation_position`, `jersey_number`,
`type_id` — so §3's `fixture_lineups` needs no extra call, and `is_starting`
is just `type_id == 11`.

---

## Q2 — What does backfilling cost? **Payload, not requests — and it's fixable.**

Request count is unchanged per fixture (lineups is an extra `include` on the
same call), but we'd have to **re-fetch all 41,777 cached fixtures**, since the
existing cache was written without the include. At ~2975 req/hr usable
(3000 minus `RESERVED_FOR_PROD`), that's **~14 hours of wall clock**.

Payload is the real problem, and the fix is server-side filtering:

Measured as we actually cache it (`json.dumps(..., indent=2)`, matching
`fetch_all_fixtures_data.py`):

| variant | per fixture | × 41,777 |
|---|---|---|
| `statistics` only (today's cache) | ~17 KiB | ~0.7 GiB |
| `lineups.details` unfiltered (60 stat types/player) | ~242 KiB | **~9.7 GiB** |
| `lineups.details` filtered to the 9 types §3 stores | ~50 KiB | ~2.0 GiB |

`filters=lineupDetailTypes:119,118,52,79,42,86,117,78,84` cuts the payload
**~4.8×** — the difference between a tractable backfill and a 10 GiB one.
(On the wire it's smaller still: ~21 KiB filtered vs ~242 KiB pretty-printed,
so writing these compact rather than indented would save several GiB again.)

**Recommendation:** backfill with an explicit detail-type filter, widened to
maybe ~15 types (the `squad_strength` extras above) rather than the bare 9 —
re-fetching 41k fixtures a second time just to add a stat type would cost
another 14 hours. Write the cache compact, not indented, at ~2 GiB either way.

---

## Q3 — How far ahead are lineups confirmed? **SportMonks predicts them ~2 days out.**

This is the finding that most changes the spec. §2 assumes "no lineup exists
until ~60–90 min before kickoff," making §5's hand-rolled `predicted_lineup()`
the main code path. That is not what the API does.

Snapshot of all 48 fixtures in the next 3 days (`/fixtures/between` +
`include=lineups`), classified by whether a bench came back — a confirmed
team sheet has one, a predicted XI is exactly 22 starters with no bench:

| hours to kickoff | confirmed | predicted | none |
|---|---|---|---|
| in play / just past | 4 | 0 | 0 |
| 6–24h | 0 | 25 | 0 |
| 24–48h | 0 | 14 | 0 |
| >48h | 0 | 0 | 5 |

So within the Upcoming tab's ~3-day window, a predicted XI exists for every
fixture inside ~48h and for none beyond it. Coverage was uniform across all
five leagues — no per-league variation in this sample, so §9.3's per-league
handling looks unnecessary for these competitions.

Two consequences:

1. **§5's `predicted_lineup()` should wrap SportMonks' predicted XI, not build
   one from start-rates.** Far less code, and a provider's prediction almost
   certainly beats a naive recent-start-rate heuristic.
2. **`lineup_source` (§5) is inferable today but only indirectly** — bench
   presence is the sole discriminator in the payload. Worth encoding as one
   documented helper rather than scattering `type_id == 12` checks.

**Not yet verified:** the exact moment a predicted XI flips to confirmed. This
snapshot had no fixture in the 0–6h band, and the four confirmed ones had
already kicked off. Phase 10's poller should log the transition and measure it,
rather than us hard-coding the spec's "60–90 minutes" guess.

---

## Q4 — Is a predicted-XI-based feature still worth it?

Can't be answered from the API; it's the Phase 9 offline comparison's job.
What Q3 changes is the shape of the question: it's no longer "team form vs. a
crude guessed XI" but "team form vs. the provider's predicted XI", which is a
much better starting point than §2 assumed. Worth noting that the
*historical training data is all confirmed lineups* while ~all serving traffic
inside 48h will be predicted XIs — a train/serve distribution gap the Phase 9
comparison needs to measure deliberately (e.g. evaluate with a simulated
predicted XI), not discover in production.

---

## Revisions this implies to the spec

- §2: drop `/squads/teams/{id}` (403) as a data source; note `sidelined` is
  fixture-include-only, no standalone endpoint.
- §2/§5: replace "lineups only ~60–90 min before kickoff" with "predicted XI
  from ~48h out, confirmed nearer kickoff"; `predicted_lineup()` becomes a
  thin wrapper over the provider's predicted XI.
- §3: `player_sidelined` is populated per fixture, so it wants a `fixture_id`
  column (and thus an as-of date) rather than being a standalone current-state
  table — otherwise there's no way to backfill it without leaking future
  information into training features.
- §4: the backfill must pass `filters=lineupDetailTypes:...`, and is a ~14h
  full re-fetch of all 41,777 cached fixtures, not an incremental add.
- §9.3: no per-league variation observed; de-prioritise.

---

## Verified: can team totals be recovered by summing player stats?

**Mostly yes, but not completely — so it can't replace the team `statistics`
include.** Summing each team's player-level `details` on fixture 19427239 and
comparing against that fixture's team `statistics`:

- **Exact match** for shots total, shots on target, shots blocked, offsides,
  fouls, passes, yellow cards, saves, total/accurate crosses, interceptions,
  duels won, key passes, big chances created, hit woodwork, goals, assists.
- **Close but not equal** for a few: tackles 14 vs 15, dribble attempts 9 vs
  11, successful dribbles 2 vs 3 — small provider inconsistencies, not a
  definitional gap.
- **Systematically off** for shots off target (15 vs 9, 5 vs 3): the team total
  appears to be defined differently from the per-player one, so this one does
  not reconcile.
- A stat absent from every player row means zero, not missing (team 20's 0
  goals yields no player rows at all) — the parser must treat absent as 0, the
  same way `extract_team_stats()` already does.

The hard limit is that **11 of the 23 `STAT_IDS` the current model trains on
have no player-level equivalent at all**: possession %, corners, red cards,
penalties, successful-passes %, long passes (62 / 27264 / 27265). Also
team-only: attacks, dangerous attacks, ball safe, shots inside/outside box,
goal kicks, free kicks, throw-ins, substitutions, successful headers, injuries.

**Implication:** player stats are a *superset in grain* but a *subset in
coverage*. Keep requesting team `statistics` alongside `lineups.details`;
summed player stats are useful as a cross-check and for stats with no team
equivalent (per-player ratings, minutes), not as a replacement.

---

## Decisions taken for the Phase 8 backfill

1. **Capture every available stat type in one pass** — no
   `lineupDetailTypes` filter. Trimming happens during feature engineering,
   where it's free; a second 14-hour re-fetch to recover a dropped type is not.
2. **Parallel cache, not an overwrite** of `data/raw/all_fixtures/`, so the
   existing model's training input is untouched and byte-identical.
3. **Store compact + gzipped.** This is what makes (1) affordable:

   | layout | per fixture | × 41,777 |
   |---|---|---|
   | pretty-printed (today's convention) | 242 KiB | 9.66 GiB |
   | compact | 115 KiB | 4.60 GiB |
   | **compact + gzip** | **9 KiB** | **0.36 GiB** |

   gzip is a ~27× win on this payload (lineup JSON is extremely repetitive),
   which makes taking all 60 stat types cheaper on disk than a filtered,
   pretty-printed cache would have been.
