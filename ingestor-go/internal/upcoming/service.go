// Package upcoming pulls not-yet-started fixtures from SportMonks within a
// forward-looking window and seeds them into Postgres, so the FastAPI
// /fixtures/upcoming endpoint (PROJECT_SPEC.md Phase 4's Upcoming tab) has
// rows to serve before kickoff.
package upcoming

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

type FixtureFetcher interface {
	FetchUpcomingBetween(ctx context.Context, start, end time.Time) ([]sportmonks.Fixture, error)
}

type FixtureStore interface {
	UpsertUpcomingFixtures(ctx context.Context, fixtures []ParsedFixture) (int, error)
}

type Service struct {
	fetcher    FixtureFetcher
	store      FixtureStore
	windowDays int
	logger     *slog.Logger
}

func NewService(fetcher FixtureFetcher, store FixtureStore, windowDays int, logger *slog.Logger) *Service {
	return &Service{
		fetcher:    fetcher,
		store:      store,
		windowDays: windowDays,
		logger:     logger,
	}
}

// Run executes pull cycles on the given interval until ctx is cancelled. It
// runs one cycle immediately rather than waiting a full interval first.
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
	start := time.Now().UTC()
	end := start.AddDate(0, 0, s.windowDays)

	s.logger.Info("pulling upcoming fixtures", "start", start.Format("2006-01-02"), "end", end.Format("2006-01-02"))

	raw, err := s.fetcher.FetchUpcomingBetween(ctx, start, end)
	if err != nil {
		if errors.Is(err, sportmonks.ErrRateLimited) {
			s.logger.Warn("skipping upcoming pull cycle: sportmonks rate limit too low")
			return
		}
		s.logger.Error("fetching upcoming fixtures failed", "error", err)
		return
	}

	parsed := make([]ParsedFixture, 0, len(raw))
	for _, f := range raw {
		pf, ok, err := parseFixture(f)
		if err != nil {
			s.logger.Warn("skipping upcoming fixture", "fixture_id", f.ID, "error", err)
			continue
		}
		if !ok {
			continue
		}
		parsed = append(parsed, pf)
	}

	changed, err := s.store.UpsertUpcomingFixtures(ctx, parsed)
	if err != nil {
		s.logger.Error("writing upcoming fixtures failed", "error", err)
		return
	}

	s.logger.Info("upcoming pull cycle complete", "fetched", len(raw), "parsed", len(parsed), "changed", changed)
}
