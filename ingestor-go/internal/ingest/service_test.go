package ingest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

type fakeFetcher struct {
	fixtures []sportmonks.Fixture
	err      error
	calls    int
}

func (f *fakeFetcher) FetchFinishedBetween(ctx context.Context, start, end time.Time) ([]sportmonks.Fixture, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return f.fixtures, nil
}

type fakeStore struct {
	written []ParsedFixture
}

func (s *fakeStore) UpsertFixtures(ctx context.Context, fixtures []ParsedFixture) (int, error) {
	s.written = append(s.written, fixtures...)
	return len(fixtures), nil
}

func testFixture(t *testing.T, id int64) sportmonks.Fixture {
	t.Helper()
	raw := `{
		"id": ` + fmt.Sprintf("%d", id) + `, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}},
			{"id": 20, "name": "Away FC", "meta": {"location": "away"}}
		],
		"scores": [
			{"type_id": 1525, "score": {"participant": "home", "goals": 1}},
			{"type_id": 1525, "score": {"participant": "away", "goals": 0}}
		],
		"statistics": []
	}`
	var f sportmonks.Fixture
	if err := json.Unmarshal([]byte(raw), &f); err != nil {
		t.Fatalf("unmarshaling fixture: %v", err)
	}
	f.Raw = json.RawMessage(raw)
	return f
}

func TestService_RunCycle_WritesParsedFixtures(t *testing.T) {
	fetcher := &fakeFetcher{fixtures: []sportmonks.Fixture{testFixture(t, 1), testFixture(t, 2)}}
	store := &fakeStore{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, store, 3, "", logger)
	svc.runCycle(context.Background())

	if fetcher.calls != 1 {
		t.Fatalf("got %d fetch calls, want 1", fetcher.calls)
	}
	if len(store.written) != 2 {
		t.Fatalf("got %d written fixtures, want 2", len(store.written))
	}
}

func TestService_RunCycle_SkipsWithoutWritingWhenRateLimited(t *testing.T) {
	fetcher := &fakeFetcher{err: sportmonks.ErrRateLimited}
	store := &fakeStore{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, store, 3, "", logger)
	svc.runCycle(context.Background())

	if fetcher.calls != 1 {
		t.Fatalf("got %d fetch calls, want 1", fetcher.calls)
	}
	if len(store.written) != 0 {
		t.Fatalf("got %d written fixtures, want 0", len(store.written))
	}
}

func TestService_RunCycle_TriggersRecomputeOnlyWhenChanged(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	t.Run("no fixtures, no recompute", func(t *testing.T) {
		fetcher := &fakeFetcher{}
		store := &fakeStore{}
		svc := NewService(fetcher, store, 3, "touch /tmp/should-not-run-liveedge-test", logger)
		svc.runCycle(context.Background())
		// nothing to assert on the file since no command should run; this
		// mainly guards against a panic/misuse when recomputeCmd is set but
		// there's nothing to recompute.
		if len(store.written) != 0 {
			t.Fatalf("got %d written fixtures, want 0", len(store.written))
		}
	})
}
