package store

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
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
