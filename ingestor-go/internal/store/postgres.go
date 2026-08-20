// Package store persists parsed fixtures to Postgres.
package store

import (
	"context"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/live"
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

-- live_match_state stands in for DynamoDB until Phase 3's AWS infra is
-- provisioned; it only ever holds each fixture's latest polled state.
CREATE TABLE IF NOT EXISTS live_match_state (
	fixture_id   BIGINT PRIMARY KEY,
	state        TEXT NOT NULL,
	home_goals   INT NOT NULL,
	away_goals   INT NOT NULL,
	updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

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

// UpsertLiveState overwrites each fixture's row with its latest polled
// state - unlike UpsertFixtures there's no history to preserve, just the
// current snapshot a DynamoDB item would hold.
func (s *Store) UpsertLiveState(ctx context.Context, states []live.MatchState) error {
	if len(states) == 0 {
		return nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("beginning transaction: %w", err)
	}
	defer tx.Rollback(ctx)

	for _, st := range states {
		if _, err := tx.Exec(ctx, `
			INSERT INTO live_match_state (fixture_id, state, home_goals, away_goals)
			VALUES ($1, $2, $3, $4)
			ON CONFLICT (fixture_id) DO UPDATE SET
				state       = EXCLUDED.state,
				home_goals  = EXCLUDED.home_goals,
				away_goals  = EXCLUDED.away_goals,
				updated_at  = now()
		`,
			st.FixtureID, st.State, st.HomeGoals, st.AwayGoals,
		); err != nil {
			return fmt.Errorf("upserting live state for fixture %d: %w", st.FixtureID, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("committing transaction: %w", err)
	}
	return nil
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
		tag, err := tx.Exec(ctx, `
			INSERT INTO odds (
				fixture_id, bookmaker_count, home_price, draw_price, away_price,
				implied_home, implied_draw, implied_away
			) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
			ON CONFLICT (fixture_id) DO UPDATE SET
				bookmaker_count = EXCLUDED.bookmaker_count,
				home_price      = EXCLUDED.home_price,
				draw_price      = EXCLUDED.draw_price,
				away_price      = EXCLUDED.away_price,
				implied_home    = EXCLUDED.implied_home,
				implied_draw    = EXCLUDED.implied_draw,
				implied_away    = EXCLUDED.implied_away,
				fetched_at      = now()
		`,
			o.FixtureID, o.BookmakerCount, o.HomePrice, o.DrawPrice, o.AwayPrice,
			o.ImpliedHome, o.ImpliedDraw, o.ImpliedAway,
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
