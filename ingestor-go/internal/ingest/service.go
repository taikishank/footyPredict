// Package ingest orchestrates pulling recently completed fixtures from
// SportMonks and persisting them, on a recurring schedule.
package ingest

import (
	"context"
	"errors"
	"log/slog"
	"os/exec"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

type FixtureFetcher interface {
	FetchFinishedBetween(ctx context.Context, start, end time.Time) ([]sportmonks.Fixture, error)
}

type FixtureStore interface {
	UpsertFixtures(ctx context.Context, fixtures []ParsedFixture) (int, error)
}

type Service struct {
	fetcher      FixtureFetcher
	store        FixtureStore
	windowDays   int
	recomputeCmd string
	logger       *slog.Logger
}

func NewService(fetcher FixtureFetcher, store FixtureStore, windowDays int, recomputeCmd string, logger *slog.Logger) *Service {
	return &Service{
		fetcher:      fetcher,
		store:        store,
		windowDays:   windowDays,
		recomputeCmd: recomputeCmd,
		logger:       logger,
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
	end := time.Now().UTC()
	start := end.AddDate(0, 0, -s.windowDays)

	s.logger.Info("pulling finished fixtures", "start", start.Format("2006-01-02"), "end", end.Format("2006-01-02"))

	raw, err := s.fetcher.FetchFinishedBetween(ctx, start, end)
	if err != nil {
		if errors.Is(err, sportmonks.ErrRateLimited) {
			s.logger.Warn("skipping pull cycle: sportmonks rate limit too low")
			return
		}
		s.logger.Error("fetching fixtures failed", "error", err)
		return
	}

	parsed := make([]ParsedFixture, 0, len(raw))
	for _, f := range raw {
		pf, ok, err := parseFixture(f)
		if err != nil {
			s.logger.Warn("skipping fixture", "fixture_id", f.ID, "error", err)
			continue
		}
		if !ok {
			continue
		}
		parsed = append(parsed, pf)
	}

	changed, err := s.store.UpsertFixtures(ctx, parsed)
	if err != nil {
		s.logger.Error("writing fixtures failed", "error", err)
		return
	}

	s.logger.Info("pull cycle complete", "fetched", len(raw), "parsed", len(parsed), "changed", changed)

	if changed > 0 && s.recomputeCmd != "" {
		s.triggerRecompute(ctx)
	}
}

func (s *Service) triggerRecompute(ctx context.Context) {
	s.logger.Info("triggering feature recompute", "cmd", s.recomputeCmd)
	cmd := exec.CommandContext(ctx, "sh", "-c", s.recomputeCmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		s.logger.Error("feature recompute failed", "error", err, "output", string(output))
		return
	}
	s.logger.Info("feature recompute finished", "output", string(output))
}
