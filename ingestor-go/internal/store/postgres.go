// Package store persists parsed fixtures to Postgres.
package store

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/live"
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
	home_goals   INT NOT NULL,
	away_goals   INT NOT NULL,
	result       TEXT NOT NULL,
	raw          JSONB NOT NULL,
	ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS fixtures_starting_at_idx ON fixtures (starting_at);

-- live_match_state stands in for DynamoDB until Phase 3's AWS infra is
-- provisioned; it only ever holds each fixture's latest polled state.
CREATE TABLE IF NOT EXISTS live_match_state (
	fixture_id   BIGINT PRIMARY KEY,
	state        TEXT NOT NULL,
	home_goals   INT NOT NULL,
	away_goals   INT NOT NULL,
	updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
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
