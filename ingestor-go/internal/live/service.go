package live

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

type Fetcher interface {
	FetchLive(ctx context.Context) ([]sportmonks.Fixture, error)
}

// StateStore holds current match state, keyed by fixture. Stands in for
// DynamoDB until Phase 3's AWS infra is provisioned.
type StateStore interface {
	UpsertLiveState(ctx context.Context, states []MatchState) error
}

// EventPublisher fans out in-match events. Stands in for SQS until Phase 3's
// AWS infra is provisioned.
type EventPublisher interface {
	PublishEvents(ctx context.Context, events []Event) error
}

type Service struct {
	fetcher   Fetcher
	states    StateStore
	publisher EventPublisher
	logger    *slog.Logger
}

func NewService(fetcher Fetcher, states StateStore, publisher EventPublisher, logger *slog.Logger) *Service {
	return &Service{
		fetcher:   fetcher,
		states:    states,
		publisher: publisher,
		logger:    logger,
	}
}

// Run executes poll cycles on the given interval until ctx is cancelled. The
// caller is responsible for choosing an interval that respects the live
// polling rate ceiling - see config.livePollInterval. It runs one cycle
// immediately rather than waiting a full interval first.
func (s *Service) Run(ctx context.Context, interval time.Duration) {
	s.runCycle(ctx)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.runCycle(ctx)
		}
	}
}

func (s *Service) runCycle(ctx context.Context) {
	raw, err := s.fetcher.FetchLive(ctx)
	if err != nil {
		if errors.Is(err, sportmonks.ErrRateLimited) {
			s.logger.Warn("skipping live poll cycle: sportmonks rate limit too low")
			return
		}
		s.logger.Error("fetching live fixtures failed", "error", err)
		return
	}

	states := make([]MatchState, 0, len(raw))
	var events []Event
	for _, f := range raw {
		state, fixtureEvents, ok, err := parseLive(f)
		if err != nil {
			s.logger.Warn("skipping live fixture", "fixture_id", f.ID, "error", err)
			continue
		}
		if !ok {
			continue
		}
		states = append(states, state)
		events = append(events, fixtureEvents...)
	}

	if len(states) > 0 {
		if err := s.states.UpsertLiveState(ctx, states); err != nil {
			s.logger.Error("writing live state failed", "error", err)
			return
		}
	}
	if len(events) > 0 {
		if err := s.publisher.PublishEvents(ctx, events); err != nil {
			s.logger.Error("publishing live events failed", "error", err)
			return
		}
	}

	s.logger.Info("live poll cycle complete", "live_fixtures", len(raw), "states_written", len(states), "events_published", len(events))
}
