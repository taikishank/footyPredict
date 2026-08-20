package store

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/upcoming"
)

// TestStore_UpsertFixtures is an integration test - it needs a live Postgres,
// so it's skipped unless TEST_POSTGRES_URL is set (e.g. the docker-compose
// instance: postgres://liveedge:liveedge@localhost:5432/liveedge?sslmode=disable).
func TestStore_UpsertFixtures(t *testing.T) {
	url := os.Getenv("TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("TEST_POSTGRES_URL not set; skipping Postgres integration test")
	}

	ctx := context.Background()
	s, err := Connect(ctx, url)
	if err != nil {
		t.Fatalf("connecting: %v", err)
	}
	defer s.Close()

	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrating: %v", err)
	}
	t.Cleanup(func() {
		s.pool.Exec(ctx, "DELETE FROM fixtures WHERE fixture_id IN (9001, 9002)")
	})

	fixture := ingest.ParsedFixture{
		FixtureID: 9001, LeagueID: 8, StartingAt: time.Now().UTC(),
		HomeID: 10, AwayID: 20, HomeName: "Home FC", AwayName: "Away FC",
		HomeGoals: 2, AwayGoals: 1, Result: "home_win",
		Raw: []byte(`{"id": 9001}`),
	}

	changed, err := s.UpsertFixtures(ctx, []ingest.ParsedFixture{fixture})
	if err != nil {
		t.Fatalf("inserting: %v", err)
	}
	if changed != 1 {
		t.Fatalf("got %d changed, want 1", changed)
	}

	// Re-upserting the identical fixture should be a no-op (raw is unchanged).
	changed, err = s.UpsertFixtures(ctx, []ingest.ParsedFixture{fixture})
	if err != nil {
		t.Fatalf("re-inserting: %v", err)
	}
	if changed != 0 {
		t.Fatalf("got %d changed on identical re-upsert, want 0", changed)
	}

	// Changing the payload should update the row.
	fixture.HomeGoals = 3
	fixture.Raw = []byte(`{"id": 9001, "updated": true}`)
	changed, err = s.UpsertFixtures(ctx, []ingest.ParsedFixture{fixture})
	if err != nil {
		t.Fatalf("updating: %v", err)
	}
	if changed != 1 {
		t.Fatalf("got %d changed on updated re-upsert, want 1", changed)
	}
}

// TestStore_UpsertUpcomingFixtures is an integration test - see
// TestStore_UpsertFixtures for the TEST_POSTGRES_URL prerequisite.
func TestStore_UpsertUpcomingFixtures(t *testing.T) {
	url := os.Getenv("TEST_POSTGRES_URL")
	if url == "" {
		t.Skip("TEST_POSTGRES_URL not set; skipping Postgres integration test")
	}

	ctx := context.Background()
	s, err := Connect(ctx, url)
	if err != nil {
		t.Fatalf("connecting: %v", err)
	}
	defer s.Close()

	if err := s.Migrate(ctx); err != nil {
		t.Fatalf("migrating: %v", err)
	}
	t.Cleanup(func() {
		s.pool.Exec(ctx, "DELETE FROM fixtures WHERE fixture_id IN (9101, 9102)")
	})

	fixture := upcoming.ParsedFixture{
		FixtureID: 9101, LeagueID: 8, StartingAt: time.Now().Add(24 * time.Hour).UTC(),
		HomeID: 10, AwayID: 20, HomeName: "Home FC", AwayName: "Away FC",
		Raw: []byte(`{"id": 9101}`),
	}

	changed, err := s.UpsertUpcomingFixtures(ctx, []upcoming.ParsedFixture{fixture})
	if err != nil {
		t.Fatalf("inserting: %v", err)
	}
	if changed != 1 {
		t.Fatalf("got %d changed, want 1", changed)
	}

	var homeGoals *int
	if err := s.pool.QueryRow(ctx, "SELECT home_goals FROM fixtures WHERE fixture_id = 9101").Scan(&homeGoals); err != nil {
		t.Fatalf("querying: %v", err)
	}
	if homeGoals != nil {
		t.Fatalf("got home_goals %v, want NULL for a not-yet-played fixture", *homeGoals)
	}

	// Once the fixture has a result (as the batch pipeline would set), the
	// upcoming pipeline must not clobber it back to NULL.
	final := ingest.ParsedFixture{
		FixtureID: 9101, LeagueID: 8, StartingAt: fixture.StartingAt,
		HomeID: 10, AwayID: 20, HomeName: "Home FC", AwayName: "Away FC",
		HomeGoals: 2, AwayGoals: 0, Result: "home_win",
		Raw: []byte(`{"id": 9101, "final": true}`),
	}
	if _, err := s.UpsertFixtures(ctx, []ingest.ParsedFixture{final}); err != nil {
		t.Fatalf("finalizing: %v", err)
	}

	changed, err = s.UpsertUpcomingFixtures(ctx, []upcoming.ParsedFixture{fixture})
	if err != nil {
		t.Fatalf("re-upserting after finalized: %v", err)
	}
	if changed != 0 {
		t.Fatalf("got %d changed, want 0 (finalized row must not be touched)", changed)
	}

	var result string
	if err := s.pool.QueryRow(ctx, "SELECT result FROM fixtures WHERE fixture_id = 9101").Scan(&result); err != nil {
		t.Fatalf("querying: %v", err)
	}
	if result != "home_win" {
		t.Fatalf("got result %q, want home_win to be preserved", result)
	}
}
