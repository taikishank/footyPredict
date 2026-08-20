package odds

import (
	"context"
	"log/slog"
	"time"
)

type OddsFetcher interface {
	FetchOdds(ctx context.Context, sportKey string) ([]Event, error)
}

type FixtureLookup interface {
	ListFixturesForLeague(ctx context.Context, leagueID int64, start, end time.Time) ([]FixtureCandidate, error)
}

type OddsStore interface {
	UpsertOdds(ctx context.Context, odds []ParsedOdds) (int, error)
}

type Service struct {
	fetcher    OddsFetcher
	lookup     FixtureLookup
	store      OddsStore
	windowDays int
	logger     *slog.Logger
}

func NewService(fetcher OddsFetcher, lookup FixtureLookup, store OddsStore, windowDays int, logger *slog.Logger) *Service {
	return &Service{
		fetcher:    fetcher,
		lookup:     lookup,
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

// runCycle fetches odds for every tracked league (see leagueSportKeys) and
// upserts whatever matches a fixture already seeded by the upcoming poller.
// One league's fetch/match failure doesn't stop the others.
func (s *Service) runCycle(ctx context.Context) {
	start := time.Now().UTC()
	end := start.AddDate(0, 0, s.windowDays)

	var parsed []ParsedOdds
	for leagueID, sportKey := range leagueSportKeys {
		candidates, err := s.lookup.ListFixturesForLeague(ctx, leagueID, start, end)
		if err != nil {
			s.logger.Error("listing fixtures for league", "league_id", leagueID, "error", err)
			continue
		}
		if len(candidates) == 0 {
			continue
		}

		events, err := s.fetcher.FetchOdds(ctx, sportKey)
		if err != nil {
			s.logger.Error("fetching odds", "sport_key", sportKey, "error", err)
			continue
		}

		matched := 0
		for _, event := range events {
			fixtureID, ok := matchFixture(event, candidates)
			if !ok {
				s.logger.Warn("no fixture match for odds event",
					"sport_key", sportKey, "home_team", event.HomeTeam, "away_team", event.AwayTeam)
				continue
			}
			po, ok := parseEvent(fixtureID, event)
			if !ok {
				s.logger.Warn("odds event has no complete h2h market", "fixture_id", fixtureID)
				continue
			}
			parsed = append(parsed, po)
			matched++
		}
		s.logger.Info("odds pull cycle for league",
			"league_id", leagueID, "candidates", len(candidates), "events", len(events), "matched", matched)
	}

	if len(parsed) == 0 {
		return
	}
	changed, err := s.store.UpsertOdds(ctx, parsed)
	if err != nil {
		s.logger.Error("writing odds failed", "error", err)
		return
	}
	s.logger.Info("odds pull cycle complete", "fixtures", len(parsed), "changed", changed)
}
