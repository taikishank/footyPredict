// Package store persists parsed fixtures to Postgres.
package store

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/odds"
	"github.com/taikishank/liveedge/ingestor-go/internal/upcoming"
)

const schema = `
CREATE TABLE IF NOT EXISTS fixtures (
	fixture_id   BIGINT PRIMARY KEY,
	league_id    BIGINT NOT NULL,
	starting_at  TIMESTAMPTZ NOT NULL,
	home_id      BIGINT NOT NULL,
	away_id      BIGINT NOT NULL,
	home_name    TEXT NOT NULL,
	away_name    TEXT NOT NULL,
	home_goals   INT,
	away_goals   INT,
	result       TEXT,
	raw          JSONB NOT NULL,
	ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fixtures_starting_at_idx ON fixtures (starting_at);

-- home_goals/away_goals/result started out NOT NULL, back when the batch
-- pipeline (finished fixtures only) was the only writer. The upcoming-
-- fixtures pipeline seeds rows before a result exists, so relax them; this
-- is a no-op on a fresh table.
ALTER TABLE fixtures ALTER COLUMN home_goals DROP NOT NULL;
ALTER TABLE fixtures ALTER COLUMN away_goals DROP NOT NULL;
ALTER TABLE fixtures ALTER COLUMN result DROP NOT NULL;

-- live_match_state moved to DynamoDB (PROJECT_SPEC.md Phase 3, see
-- internal/live/dynamo_store.go) - it was always latest-snapshot-only with
-- no history, so there's nothing worth migrating out of this table.
DROP TABLE IF EXISTS live_match_state;

-- odds holds each fixture's latest de-vigged bookmaker prices (PROJECT_SPEC.md
-- Phase 4). Like live_match_state, it's latest-only - the odds poller
-- overwrites rather than accumulates history.
CREATE TABLE IF NOT EXISTS odds (
	fixture_id      BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id),
	bookmaker_count INT NOT NULL,
	home_price      NUMERIC NOT NULL,
	draw_price      NUMERIC NOT NULL,
	away_price      NUMERIC NOT NULL,
	implied_home    NUMERIC NOT NULL,
	implied_draw    NUMERIC NOT NULL,
	implied_away    NUMERIC NOT NULL,
	fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- btts is a second, independent market on the same fixture (PROJECT_SPEC.md's
-- BTTS phase), added as columns on the same latest-only snapshot rather than
-- a second table: one row per fixture either way, and a fixture can have h2h
-- without btts (or vice versa) on a given poll, hence all nullable.
ALTER TABLE odds ADD COLUMN IF NOT EXISTS btts_bookmaker_count INT;
ALTER TABLE odds ADD COLUMN IF NOT EXISTS btts_yes_price NUMERIC;
ALTER TABLE odds ADD COLUMN IF NOT EXISTS btts_no_price NUMERIC;
ALTER TABLE odds ADD COLUMN IF NOT EXISTS implied_btts_yes NUMERIC;
ALTER TABLE odds ADD COLUMN IF NOT EXISTS implied_btts_no NUMERIC;

-- edges is append-only, unlike odds/live_match_state: inference-py inserts a
-- row each time it serves /fixtures/upcoming for a fixture with an odds
-- match, so Phase 6's backtest module has real history to grade against
-- actual outcomes rather than only ever seeing the latest snapshot.
CREATE TABLE IF NOT EXISTS edges (
	fixture_id   BIGINT NOT NULL REFERENCES fixtures(fixture_id),
	computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
	model_home   NUMERIC NOT NULL,
	model_draw   NUMERIC NOT NULL,
	model_away   NUMERIC NOT NULL,
	market_home  NUMERIC NOT NULL,
	market_draw  NUMERIC NOT NULL,
	market_away  NUMERIC NOT NULL,
	edge_home    NUMERIC NOT NULL,
	edge_draw    NUMERIC NOT NULL,
	edge_away    NUMERIC NOT NULL,
	flagged      BOOLEAN NOT NULL,
	PRIMARY KEY (fixture_id, computed_at)
);

-- Prices at the moment the edge was computed, so a graded prediction_call
-- can simulate P&L against what was actually available rather than the
-- (possibly since-moved) closing price in odds. Added after edges shipped,
-- hence ALTER rather than inline columns; nullable since older rows predate
-- this and will just be excluded from P&L.
ALTER TABLE edges ADD COLUMN IF NOT EXISTS home_price NUMERIC;
ALTER TABLE edges ADD COLUMN IF NOT EXISTS draw_price NUMERIC;
ALTER TABLE edges ADD COLUMN IF NOT EXISTS away_price NUMERIC;

-- Which model produced the snapshot. inference-py served v1 until the
-- switch to the v2 challenger (team form + club_goals_relative), so DEFAULT
-- 'v1' correctly labels every pre-existing row; the default is then dropped
-- so a writer that forgets the column fails loudly instead of silently
-- mislabelling v2 rows as v1. Only inference-py inserts here.
ALTER TABLE edges ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'v1';
ALTER TABLE edges ALTER COLUMN model_version DROP DEFAULT;

-- prediction_calls is the graded scorecard: one row per fixture the model
-- ever priced, selecting the edges snapshot closest to kickoff as "the call"
-- (see backtest/grade_calls.py) and recording what actually happened. edges
-- stays a pure append-only log; this is derived from it once a fixture's
-- result is known.
CREATE TABLE IF NOT EXISTS prediction_calls (
	fixture_id      BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id),
	edge_computed_at TIMESTAMPTZ NOT NULL,
	call_side       TEXT NOT NULL CHECK (call_side IN ('home_win', 'draw', 'away_win')),
	model_prob      NUMERIC NOT NULL,
	market_prob     NUMERIC NOT NULL,
	edge            NUMERIC NOT NULL,
	price           NUMERIC,
	actual_result   TEXT NOT NULL CHECK (actual_result IN ('home_win', 'draw', 'away_win')),
	correct         BOOLEAN NOT NULL,
	brier_score     NUMERIC NOT NULL,
	market_brier_score NUMERIC NOT NULL,
	staked_units    NUMERIC NOT NULL DEFAULT 1,
	pnl_units       NUMERIC,
	graded_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
	FOREIGN KEY (fixture_id, edge_computed_at) REFERENCES edges (fixture_id, computed_at)
);

-- Every priced fixture is graded, not just the ones the model flagged an
-- edge on, so calibration is measured on the whole book rather than on a
-- self-selected slice. flagged is the state of the graded snapshot itself;
-- ever_flagged is true if ANY pre-kickoff snapshot was flagged, since a
-- fixture can flip in and out of flagged as odds move. Added after the
-- flagged-only grading shipped, hence ALTER; DEFAULT true is correct for
-- the pre-existing rows, which were all flagged calls by construction.
ALTER TABLE prediction_calls ADD COLUMN IF NOT EXISTS flagged BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE prediction_calls ADD COLUMN IF NOT EXISTS ever_flagged BOOLEAN NOT NULL DEFAULT true;

-- Carried over from the graded edges snapshot, so a report can never mix
-- two models' records. The primary key widens to include it: one fixture
-- can legitimately be graded once per model version.
ALTER TABLE prediction_calls ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'v1';
ALTER TABLE prediction_calls ALTER COLUMN model_version DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'prediction_calls'::regclass AND i.indisprimary
		  AND a.attname = 'model_version'
	) THEN
		ALTER TABLE prediction_calls DROP CONSTRAINT prediction_calls_pkey;
		ALTER TABLE prediction_calls ADD PRIMARY KEY (fixture_id, model_version);
	END IF;
END $$;

-- double_chance_calls is prediction_calls' binary-market sibling: for the
-- same flagged edge snapshot, grades the double-chance bet that excludes
-- whichever outcome prediction_calls.call_side names (e.g. call_side
-- 'away_win' -> dc_label '1X', home-or-draw). Derived entirely from the
-- same snapshot/actual result, so it needs no new odds data - see
-- backtest/grade_calls.py.
CREATE TABLE IF NOT EXISTS double_chance_calls (
	fixture_id       BIGINT PRIMARY KEY REFERENCES fixtures(fixture_id),
	edge_computed_at TIMESTAMPTZ NOT NULL,
	dc_label         TEXT NOT NULL CHECK (dc_label IN ('1X', 'X2', '12')),
	excludes_outcome TEXT NOT NULL CHECK (excludes_outcome IN ('home_win', 'draw', 'away_win')),
	model_prob       NUMERIC NOT NULL,
	market_prob      NUMERIC NOT NULL,
	edge             NUMERIC NOT NULL,
	implied_price    NUMERIC,
	actual_result    TEXT NOT NULL CHECK (actual_result IN ('home_win', 'draw', 'away_win')),
	correct          BOOLEAN NOT NULL,
	brier_score      NUMERIC NOT NULL,
	market_brier_score NUMERIC NOT NULL,
	staked_units     NUMERIC NOT NULL DEFAULT 1,
	pnl_units        NUMERIC,
	graded_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
	FOREIGN KEY (fixture_id, edge_computed_at) REFERENCES edges (fixture_id, computed_at)
);

-- Same flagged/ever_flagged split as prediction_calls, for the same reason.
ALTER TABLE double_chance_calls ADD COLUMN IF NOT EXISTS flagged BOOLEAN NOT NULL DEFAULT true;
ALTER TABLE double_chance_calls ADD COLUMN IF NOT EXISTS ever_flagged BOOLEAN NOT NULL DEFAULT true;

-- Same model tag and widened key as prediction_calls, for the same reason.
ALTER TABLE double_chance_calls ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'v1';
ALTER TABLE double_chance_calls ALTER COLUMN model_version DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'double_chance_calls'::regclass AND i.indisprimary
		  AND a.attname = 'model_version'
	) THEN
		ALTER TABLE double_chance_calls DROP CONSTRAINT double_chance_calls_pkey;
		ALTER TABLE double_chance_calls ADD PRIMARY KEY (fixture_id, model_version);
	END IF;
END $$;

-- edges' key widens the same way, and for the same reason - it just took
-- longer to notice. The key predates model_version, so it is
-- (fixture_id, computed_at): one snapshot per fixture per instant, whoever
-- wrote it. That was true while only inference-py inserted here and only one
-- model served. It stopped being true when backtest/backfill_predictions.py
-- began reconstructing a second model's snapshots over already-played
-- fixtures, because both models derive computed_at from the same odds row's
-- fetched_at and therefore collide on the key. The loser was silently
-- discarded by that script's ON CONFLICT DO NOTHING: a v3 reconstruction of
-- 43 fixtures wrote 12 and reported 43.
--
-- Both child tables already carry model_version and already key on it, so the
-- two composite foreign keys widen with the parent rather than being dropped.
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'edges'::regclass AND i.indisprimary
		  AND a.attname = 'model_version'
	) THEN
		ALTER TABLE prediction_calls
			DROP CONSTRAINT prediction_calls_fixture_id_edge_computed_at_fkey;
		ALTER TABLE double_chance_calls
			DROP CONSTRAINT double_chance_calls_fixture_id_edge_computed_at_fkey;

		ALTER TABLE edges DROP CONSTRAINT edges_pkey;
		ALTER TABLE edges ADD PRIMARY KEY (fixture_id, computed_at, model_version);

		ALTER TABLE prediction_calls
			ADD CONSTRAINT prediction_calls_fixture_id_edge_computed_at_fkey
			FOREIGN KEY (fixture_id, edge_computed_at, model_version)
			REFERENCES edges (fixture_id, computed_at, model_version);
		ALTER TABLE double_chance_calls
			ADD CONSTRAINT double_chance_calls_fixture_id_edge_computed_at_fkey
			FOREIGN KEY (fixture_id, edge_computed_at, model_version)
			REFERENCES edges (fixture_id, computed_at, model_version);
	END IF;
END $$;

-- Which selection rule turned the snapshot's probabilities into this bet
-- (backtest/selection.py). model_version and call_rule are different axes:
-- the first says which artifact produced model_home/draw/away, the second
-- what was done with them. Every row written before the split was following
-- max_edge, which is what DEFAULT 'max_edge' labels them; the default is
-- then dropped so a writer that forgets the column fails loudly rather than
-- silently mislabelling a belief-arm row as an edge-arm one - the same
-- reason model_version drops its default above.
--
-- The key widens again: one fixture is legitimately graded once per model
-- version PER RULE, and the two rows deliberately share an edges snapshot,
-- so the snapshot foreign key must NOT widen with it.
ALTER TABLE prediction_calls ADD COLUMN IF NOT EXISTS call_rule TEXT NOT NULL DEFAULT 'max_edge';
ALTER TABLE prediction_calls ALTER COLUMN call_rule DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'prediction_calls'::regclass AND i.indisprimary
		  AND a.attname = 'call_rule'
	) THEN
		ALTER TABLE prediction_calls DROP CONSTRAINT prediction_calls_pkey;
		ALTER TABLE prediction_calls ADD PRIMARY KEY (fixture_id, model_version, call_rule);
	END IF;
END $$;

-- Same column and same widened key on the DC arm, for the same reason. The
-- DC arm selects its own side, so it has its own rule of the same name
-- rather than inheriting the singles pick (see backtest/selection.py).
ALTER TABLE double_chance_calls ADD COLUMN IF NOT EXISTS call_rule TEXT NOT NULL DEFAULT 'max_edge';
ALTER TABLE double_chance_calls ALTER COLUMN call_rule DROP DEFAULT;
DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1 FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
		WHERE i.indrelid = 'double_chance_calls'::regclass AND i.indisprimary
		  AND a.attname = 'call_rule'
	) THEN
		ALTER TABLE double_chance_calls DROP CONSTRAINT double_chance_calls_pkey;
		ALTER TABLE double_chance_calls ADD PRIMARY KEY (fixture_id, model_version, call_rule);
	END IF;
END $$;

-- edges_btts is the BTTS market's edges: append-only, same as edges, and
-- keyed the same final shape edges only reached after two migrations above -
-- (fixture_id, computed_at, model_version) - since this table starts fresh
-- rather than growing into it. Two outcomes (yes/no), not three, is the only
-- real difference from edges.
CREATE TABLE IF NOT EXISTS edges_btts (
	fixture_id    BIGINT NOT NULL REFERENCES fixtures(fixture_id),
	computed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
	model_yes     NUMERIC NOT NULL,
	model_no      NUMERIC NOT NULL,
	market_yes    NUMERIC NOT NULL,
	market_no     NUMERIC NOT NULL,
	edge_yes      NUMERIC NOT NULL,
	edge_no       NUMERIC NOT NULL,
	flagged       BOOLEAN NOT NULL,
	yes_price     NUMERIC,
	no_price      NUMERIC,
	model_version TEXT NOT NULL,
	PRIMARY KEY (fixture_id, computed_at, model_version)
);

-- btts_calls is edges_btts' graded scorecard, prediction_calls' BTTS sibling -
-- same (fixture_id, model_version, call_rule) key prediction_calls only
-- reached after its own call_rule migration above, started here directly.
-- actual_btts is read straight off fixtures.home_goals/away_goals
-- (backtest/grade_calls.py), not off the categorical result column
-- prediction_calls uses, since "home_win/draw/away_win" doesn't carry
-- whether either side was shut out.
CREATE TABLE IF NOT EXISTS btts_calls (
	fixture_id         BIGINT NOT NULL REFERENCES fixtures(fixture_id),
	edge_computed_at   TIMESTAMPTZ NOT NULL,
	call_rule          TEXT NOT NULL,
	call_side          TEXT NOT NULL CHECK (call_side IN ('yes', 'no')),
	model_prob         NUMERIC NOT NULL,
	market_prob        NUMERIC NOT NULL,
	edge               NUMERIC NOT NULL,
	price              NUMERIC,
	actual_btts        BOOLEAN NOT NULL,
	correct            BOOLEAN NOT NULL,
	brier_score        NUMERIC NOT NULL,
	market_brier_score NUMERIC NOT NULL,
	staked_units       NUMERIC NOT NULL DEFAULT 1,
	pnl_units          NUMERIC,
	flagged            BOOLEAN NOT NULL,
	ever_flagged       BOOLEAN NOT NULL,
	model_version      TEXT NOT NULL,
	graded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	PRIMARY KEY (fixture_id, model_version, call_rule),
	FOREIGN KEY (fixture_id, edge_computed_at, model_version)
		REFERENCES edges_btts (fixture_id, computed_at, model_version)
);
`

type Store struct {
	pool *pgxpool.Pool
}

func Connect(ctx context.Context, url string) (*Store, error) {
	pool, err := pgxpool.New(ctx, url)
	if err != nil {
		return nil, fmt.Errorf("connecting to postgres: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("pinging postgres: %w", err)
	}
	return &Store{pool: pool}, nil
}

func (s *Store) Close() {
	s.pool.Close()
}

// Migrate creates the fixtures table if it doesn't already exist. Good
// enough for this phase; a real migration tool (goose/migrate) can replace
// it once the schema needs versioned changes.
func (s *Store) Migrate(ctx context.Context) error {
	if _, err := s.pool.Exec(ctx, schema); err != nil {
		return fmt.Errorf("running migration: %w", err)
	}
	return nil
}

// UpsertFixtures writes parsed fixtures, overwriting any existing row for
// the same fixture_id (a fixture's stats can still be corrected/finalized
// shortly after full time). It returns how many rows were inserted or
// changed, so callers can decide whether a feature recompute is warranted.
func (s *Store) UpsertFixtures(ctx context.Context, fixtures []ingest.ParsedFixture) (int, error) {
	if len(fixtures) == 0 {
		return 0, nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	changed := 0
	for _, f := range fixtures {
		tag, err := tx.Exec(ctx, `
			INSERT INTO fixtures (
				fixture_id, league_id, starting_at, home_id, away_id,
				home_name, away_name, home_goals, away_goals, result, raw
			) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
			ON CONFLICT (fixture_id) DO UPDATE SET
				home_goals = EXCLUDED.home_goals,
				away_goals = EXCLUDED.away_goals,
				result     = EXCLUDED.result,
				raw        = EXCLUDED.raw,
				ingested_at = now()
			WHERE fixtures.raw IS DISTINCT FROM EXCLUDED.raw
		`,
			f.FixtureID, f.LeagueID, f.StartingAt, f.HomeID, f.AwayID,
			f.HomeName, f.AwayName, f.HomeGoals, f.AwayGoals, f.Result, f.Raw,
		)
		if err != nil {
			return 0, fmt.Errorf("upserting fixture %d: %w", f.FixtureID, err)
		}
		changed += int(tag.RowsAffected())
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("committing transaction: %w", err)
	}
	return changed, nil
}

// OldestUnresolvedStartingAt returns the starting_at of the oldest fixture
// that has already kicked off (starting_at < before) but never got a result
// backfilled (home_goals IS NULL) - i.e. a gap UpsertFixtures's rolling
// window has already moved past. Returns nil if there's no such gap.
func (s *Store) OldestUnresolvedStartingAt(ctx context.Context, before time.Time) (*time.Time, error) {
	var startingAt time.Time
	err := s.pool.QueryRow(ctx, `
		SELECT starting_at FROM fixtures
		WHERE starting_at < $1 AND home_goals IS NULL
		ORDER BY starting_at ASC
		LIMIT 1
	`, before).Scan(&startingAt)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("querying oldest unresolved fixture: %w", err)
	}
	return &startingAt, nil
}

// UpsertUpcomingFixtures seeds fixtures rows ahead of kickoff, with no
// score/result yet. It never overwrites a row that already has a result -
// once the batch pipeline (UpsertFixtures) has finalized a fixture, this
// pipeline backs off rather than re-fetching a fixture that's already
// dropped out of its own forward-looking window.
func (s *Store) UpsertUpcomingFixtures(ctx context.Context, fixtures []upcoming.ParsedFixture) (int, error) {
	if len(fixtures) == 0 {
		return 0, nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	changed := 0
	for _, f := range fixtures {
		tag, err := tx.Exec(ctx, `
			INSERT INTO fixtures (
				fixture_id, league_id, starting_at, home_id, away_id,
				home_name, away_name, raw
			) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (fixture_id) DO UPDATE SET
				league_id   = EXCLUDED.league_id,
				starting_at = EXCLUDED.starting_at,
				home_id     = EXCLUDED.home_id,
				away_id     = EXCLUDED.away_id,
				home_name   = EXCLUDED.home_name,
				away_name   = EXCLUDED.away_name,
				raw         = EXCLUDED.raw,
				ingested_at = now()
			WHERE fixtures.result IS NULL AND fixtures.raw IS DISTINCT FROM EXCLUDED.raw
		`,
			f.FixtureID, f.LeagueID, f.StartingAt, f.HomeID, f.AwayID,
			f.HomeName, f.AwayName, f.Raw,
		)
		if err != nil {
			return 0, fmt.Errorf("upserting upcoming fixture %d: %w", f.FixtureID, err)
		}
		changed += int(tag.RowsAffected())
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("committing transaction: %w", err)
	}
	return changed, nil
}

// ListFixturesForLeague returns fixtures in the given league kicking off in
// [start, end), for the odds poller to match The Odds API's events against
// (see odds.Service.runCycle).
func (s *Store) ListFixturesForLeague(ctx context.Context, leagueID int64, start, end time.Time) ([]odds.FixtureCandidate, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT fixture_id, home_name, away_name, starting_at
		FROM fixtures
		WHERE league_id = $1 AND starting_at >= $2 AND starting_at < $3
	`, leagueID, start, end)
	if err != nil {
		return nil, fmt.Errorf("listing fixtures for league %d: %w", leagueID, err)
	}
	defer rows.Close()

	var candidates []odds.FixtureCandidate
	for rows.Next() {
		var c odds.FixtureCandidate
		if err := rows.Scan(&c.FixtureID, &c.HomeName, &c.AwayName, &c.StartingAt); err != nil {
			return nil, fmt.Errorf("scanning fixture candidate: %w", err)
		}
		candidates = append(candidates, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("reading fixture candidates: %w", err)
	}
	return candidates, nil
}

// UpsertOdds overwrites each fixture's row with its latest de-vigged prices -
// like UpsertLiveState, this is a snapshot table with no history.
func (s *Store) UpsertOdds(ctx context.Context, parsed []odds.ParsedOdds) (int, error) {
	if len(parsed) == 0 {
		return 0, nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	changed := 0
	for _, o := range parsed {
		// btts is NULLed out, not left stale, when this poll's event had no
		// complete btts market - a missing quote this cycle should not read
		// as a surviving one from whenever btts was last seen.
		var bttsCount *int
		var bttsYes, bttsNo, impliedYes, impliedNo *float64
		if o.HasBTTS {
			bttsCount = &o.BTTSBookmakerCount
			bttsYes, bttsNo = &o.BTTSYesPrice, &o.BTTSNoPrice
			impliedYes, impliedNo = &o.ImpliedBTTSYes, &o.ImpliedBTTSNo
		}

		tag, err := tx.Exec(ctx, `
			INSERT INTO odds (
				fixture_id, bookmaker_count, home_price, draw_price, away_price,
				implied_home, implied_draw, implied_away,
				btts_bookmaker_count, btts_yes_price, btts_no_price,
				implied_btts_yes, implied_btts_no
			) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
			ON CONFLICT (fixture_id) DO UPDATE SET
				bookmaker_count      = EXCLUDED.bookmaker_count,
				home_price           = EXCLUDED.home_price,
				draw_price           = EXCLUDED.draw_price,
				away_price           = EXCLUDED.away_price,
				implied_home         = EXCLUDED.implied_home,
				implied_draw         = EXCLUDED.implied_draw,
				implied_away         = EXCLUDED.implied_away,
				btts_bookmaker_count = EXCLUDED.btts_bookmaker_count,
				btts_yes_price       = EXCLUDED.btts_yes_price,
				btts_no_price        = EXCLUDED.btts_no_price,
				implied_btts_yes     = EXCLUDED.implied_btts_yes,
				implied_btts_no      = EXCLUDED.implied_btts_no,
				fetched_at           = now()
		`,
			o.FixtureID, o.BookmakerCount, o.HomePrice, o.DrawPrice, o.AwayPrice,
			o.ImpliedHome, o.ImpliedDraw, o.ImpliedAway,
			bttsCount, bttsYes, bttsNo, impliedYes, impliedNo,
		)
		if err != nil {
			return 0, fmt.Errorf("upserting odds for fixture %d: %w", o.FixtureID, err)
		}
		changed += int(tag.RowsAffected())
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, fmt.Errorf("committing transaction: %w", err)
	}
	return changed, nil
}
