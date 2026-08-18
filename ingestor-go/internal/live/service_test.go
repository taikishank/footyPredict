package live

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"testing"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

type fakeFetcher struct {
	fixtures []sportmonks.Fixture
	err      error
	calls    int
}

func (f *fakeFetcher) FetchLive(ctx context.Context) ([]sportmonks.Fixture, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	return f.fixtures, nil
}

type fakeStates struct {
	written []MatchState
	err     error
}

func (s *fakeStates) UpsertLiveState(ctx context.Context, states []MatchState) error {
	if s.err != nil {
		return s.err
	}
	s.written = append(s.written, states...)
	return nil
}

type fakePublisher struct {
	published []Event
}

func (p *fakePublisher) PublishEvents(ctx context.Context, events []Event) error {
	p.published = append(p.published, events...)
	return nil
}

func testLiveFixture(t *testing.T, id int64) sportmonks.Fixture {
	t.Helper()
	raw := `{"id": ` + fmt.Sprintf("%d", id) + `, "state": {"short_name": "1H"}, "scores": [
		{"type_id": 1525, "score": {"participant": "home", "goals": 1}},
		{"type_id": 1525, "score": {"participant": "away", "goals": 0}}
	], "events": [
		{"id": 1, "type_id": 14, "participant_id": 10, "player_name": "Scorer", "minute": 20}
	]}`
	var f sportmonks.Fixture
	if err := json.Unmarshal([]byte(raw), &f); err != nil {
		t.Fatalf("unmarshaling fixture: %v", err)
	}
	return f
}

func TestService_RunCycle_WritesStateAndPublishesEvents(t *testing.T) {
	fetcher := &fakeFetcher{fixtures: []sportmonks.Fixture{testLiveFixture(t, 1), testLiveFixture(t, 2)}}
	states := &fakeStates{}
	publisher := &fakePublisher{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, states, publisher, logger)
	svc.runCycle(context.Background())

	if fetcher.calls != 1 {
		t.Fatalf("got %d fetch calls, want 1", fetcher.calls)
	}
	if len(states.written) != 2 {
		t.Fatalf("got %d states written, want 2", len(states.written))
	}
	if len(publisher.published) != 2 {
		t.Fatalf("got %d events published, want 2", len(publisher.published))
	}
}

func TestService_RunCycle_SkipsWithoutWritingWhenRateLimited(t *testing.T) {
	fetcher := &fakeFetcher{err: sportmonks.ErrRateLimited}
	states := &fakeStates{}
	publisher := &fakePublisher{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, states, publisher, logger)
	svc.runCycle(context.Background())

	if len(states.written) != 0 || len(publisher.published) != 0 {
		t.Fatalf("expected nothing written or published, got states=%d events=%d", len(states.written), len(publisher.published))
	}
}

func TestService_RunCycle_StopsBeforePublishingWhenStateWriteFails(t *testing.T) {
	fetcher := &fakeFetcher{fixtures: []sportmonks.Fixture{testLiveFixture(t, 1)}}
	states := &fakeStates{err: errors.New("boom")}
	publisher := &fakePublisher{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, states, publisher, logger)
	svc.runCycle(context.Background())

	if len(publisher.published) != 0 {
		t.Fatalf("expected no events published after state write failure, got %d", len(publisher.published))
	}
}
