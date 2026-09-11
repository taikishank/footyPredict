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

---

## Historical depth: lineups go back to 2004, per-player stats only to 2015/16

Lineups being *present* historically is not the same as being *useful*. Sampling
fixtures per year (`coverage_by_season.py`, evenly spread across each year and
split by league so one league's gap can't read as a league-wide cliff):

| era | distinct stat types | stat rows per named player |
|---|---|---|
| 2004–2005 | 0 | 0 — team sheets only |
| 2006–2014 | 5–8 | ~1.5 |
| 2015 (transition) | 35–43 | ~5 |
| 2016–2020 | 37–48 | ~13 |
| 2021–2024 | 42–63 | ~13 |
| 2025–2026 | 64–65 | ~20 |

Full 35–40-man team sheets come back from 2004 onward; it's the per-player
*stats* that are thin. The jump is uniform across all five leagues, so it's a
provider-wide reporting change at the **2015/16 season** boundary, not a gap in
one competition.

Of 40,787 already-played cached fixtures, **19,756 (48%) fall in the dense
2016+ era**. Backfilling only those would take ~6.6h instead of ~13.7h. We're
backfilling everything anyway — the thin seasons cost little at ~4 KiB each,
and which features the sparse era actually loses is a feature-engineering
question, better answered with the data in hand than by guessing now.

---

## Season completeness: every season is whole except four

`check_season_completeness.py` checks each league-season in the cache, with no
API calls. Three checks: season size against the double round-robin its team
count implies, per-team fixture counts, and per-round fixture counts.

The first version of this check got it wrong in a way worth recording: it
counted *cached fixture rows*, and a cached row is not evidence a match was
played. Ligue 1 2019/20 has all 380 rows, 101 of them Cancelled or Postponed —
it read as complete. Completeness is now counted over played fixtures only
(states 5 Full Time, 7 After Extra Time), treating 10 Postponed, 12 Cancelled,
15 Abandoned and 17 Awarded as never played.

**115 league-seasons: 106 complete (3 of them with uneven rounds), 4 with holes,
5 not yet started in the cache.**

| season | played / expected | cause | per-team impact |
|---|---|---|---|
| Ligue 1 2019/20 | 279 / 380 | 80 cancelled, 21 postponed (COVID) | every team short 10–11 |
| Ligue 1 2024/25 | 305 / 306 | 1 abandoned | 2 teams short 1 |
| Bundesliga 2024/25 | 305 / 306 | 1 awarded | 2 teams short 1 |
| Ligue 1 2025/26 | 305 / 306 | 1 abandoned | 2 teams short 1 |

**Recommendation: omit Ligue 1 2019/20 entirely.** Every team lost 10–11
fixtures, so every rolling form window in that season silently spans a
three-month void. The other three are one missing match each, affecting two
teams' windows for a few fixtures — worth knowing, not worth dropping a season
over, though the simplest safe handling is to drop the affected teams' fixtures
immediately following the abandoned match.

No season has a team quietly short by a handful of fixtures beyond these, which
was the specific failure mode worth hunting: too few absences to notice in
aggregate, enough to distort a team's form features.

### Separate finding: the offline fixture cache is stale for 2026/27

The five current-season league-seasons come back as "not started" in their
entirety, which is a cache artifact, not reality: **146 fixtures across the five
leagues have a kickoff already in the past but are cached as Not Started**, with
no result and no statistics (Premier League 30, La Liga 41, Serie A 30, Ligue 1
27, Bundesliga 18). Spot-checked against the API - e.g. fixture 19722203
(2026-08-21) is cached `state_id: 1` but live is Full Time.

This affects the **incumbent** model too, not just this project:
`data/raw/all_fixtures/` is what `build_features.py` reads, so the 2026/27
season to date is absent from offline training. Re-running
`fetch_all_fixtures_data.py` won't fix it either - it skips any fixture already
cached, so these 146 stale files are never refetched. Worth a targeted refresh
of fixtures whose cached state is Not Started but whose kickoff has passed.

### A false alarm worth not re-raising

Three complete seasons (Premier League 2005/06 and 2007/08, La Liga 2005/06)
have unevenly-sized rounds — a round with 6–9 fixtures where 10 were expected.
Nothing is missing: postponed matches were reassigned to a different
`round_id`, leaving one round short and another long. Since the season total and
every per-team count are correct, these are reported as informational rather
than alerts. Round *numbers* aren't in the payload at all, only ids, so rounds
are ordered by first kickoff and identified by position.
