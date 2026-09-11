-- Phase 8 tables for PLAYER_STATS_SPEC.md section 3, revised by what the
-- availability probe actually found (FINDINGS_DATA_AVAILABILITY.md).
--
-- Kept as standalone SQL for now rather than appended to the `schema` string in
-- ingestor-go/internal/store/postgres.go: nothing writes these until Phase 10's
-- lineup poller exists, and folding them in early would run migrations in
-- production for tables no code touches yet. When the Go writer lands, this
-- moves there verbatim, following the same CREATE TABLE IF NOT EXISTS /
-- ALTER TABLE ... ADD COLUMN IF NOT EXISTS convention.
--
-- Two departures from the spec, both deliberate:
--
-- 1. player_fixture_stats is LONG - one row per (fixture, player, stat type) -
--    not the ~10 wide columns the spec sketched. The payload carries ~60 stat
--    types per player and the backfill keeps all of them, so wide would mean a
--    migration per new stat. Long is one schema forever and pivots cheaply at
--    feature-build time.
-- 2. player_sidelined is keyed by fixture, not "current". The spec's version is
--    a standalone current-state table, but our plan exposes sidelined only as a
--    per-fixture include (no /squads, no standalone /sidelined endpoint), and a
--    current-state table cannot be reconstructed historically without leaking
--    information that was not known at kickoff.

CREATE TABLE IF NOT EXISTS players (
    player_id     BIGINT PRIMARY KEY,
    name          TEXT,
    position      TEXT,          -- goalkeeper/defender/midfielder/attacker
    date_of_birth DATE,
    raw           JSONB
);

-- type_id -> name, so the long stats table is readable without a lookup in
-- application code. Seeded from ml/data/raw/player_stats_probe/core_types.json.
CREATE TABLE IF NOT EXISTS stat_types (
    type_id BIGINT PRIMARY KEY,
    name    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fixture_lineups (
    fixture_id        BIGINT NOT NULL REFERENCES fixtures(fixture_id),
    team_id           BIGINT NOT NULL,
    player_id         BIGINT NOT NULL REFERENCES players(player_id),
    is_starting       BOOLEAN NOT NULL,
    position          TEXT,
    position_id       BIGINT,
    formation_field   TEXT,
    formation_position INT,
    jersey_number     INT,
    -- 'confirmed' or 'predicted': SportMonks serves a predicted XI from ~48h
    -- out, and a prediction built on one is materially weaker than one built on
    -- a real team sheet, so the distinction has to survive into serving
    -- (spec section 5's lineup_source). Inferred from bench presence.
    lineup_source     TEXT NOT NULL CHECK (lineup_source IN ('confirmed', 'predicted')),
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (fixture_id, player_id)
);
CREATE INDEX IF NOT EXISTS fixture_lineups_player_idx ON fixture_lineups (player_id);
CREATE INDEX IF NOT EXISTS fixture_lineups_team_idx ON fixture_lineups (team_id);

-- Long form: one row per (fixture, player, stat type). An absent row means
-- zero, not unknown - the same contract extract_team_stats() already uses for
-- team stats - so readers must not treat a missing row as missing data.
CREATE TABLE IF NOT EXISTS player_fixture_stats (
    fixture_id BIGINT NOT NULL REFERENCES fixtures(fixture_id),
    player_id  BIGINT NOT NULL REFERENCES players(player_id),
    team_id    BIGINT NOT NULL,
    type_id    BIGINT NOT NULL REFERENCES stat_types(type_id),
    value      NUMERIC,   -- the numeric case: counts, ratings, percentages
    raw_value  JSONB,     -- anything non-numeric (e.g. Captain arrives as a bool)
    PRIMARY KEY (fixture_id, player_id, type_id)
);
-- player_form() (spec section 5) windows by a player's own appearances, so the
-- hot path is "all stats for one player, most recent first".
CREATE INDEX IF NOT EXISTS player_fixture_stats_player_idx
    ON player_fixture_stats (player_id, fixture_id);
CREATE INDEX IF NOT EXISTS player_fixture_stats_type_idx
    ON player_fixture_stats (type_id);

CREATE TABLE IF NOT EXISTS player_sidelined (
    fixture_id     BIGINT NOT NULL REFERENCES fixtures(fixture_id),
    sideline_id    BIGINT NOT NULL,
    player_id      BIGINT NOT NULL REFERENCES players(player_id),
    team_id        BIGINT NOT NULL,
    category       TEXT,    -- injury / suspension / ...
    reason_type_id BIGINT,  -- e.g. 537 'Ankle Injury'
    start_date     DATE,
    end_date       DATE,
    games_missed   INT,
    raw            JSONB,
    -- one player can carry several sideline records into the same fixture
    PRIMARY KEY (fixture_id, sideline_id)
);
CREATE INDEX IF NOT EXISTS player_sidelined_player_idx ON player_sidelined (player_id);
