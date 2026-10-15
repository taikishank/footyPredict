package odds

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"
)

type fakeFetcher struct {
	events     map[string][]Event // keyed by sport_key
	bttsEvents map[string]Event   // keyed by event id; absent id = no btts quotes
	calls      int
}

func (f *fakeFetcher) FetchOdds(ctx context.Context, sportKey string) ([]Event, error) {
	f.calls++
	return f.events[sportKey], nil
}

// FetchEventBTTS returns the configured per-event btts quotes, or an empty
// Event (no bookmakers, matching a fixture nobody has quoted btts for yet)
// when bttsEvents has nothing for this id.
func (f *fakeFetcher) FetchEventBTTS(ctx context.Context, sportKey, eventID string) (Event, error) {
	return f.bttsEvents[eventID], nil
}

type fakeLookup struct {
	candidates map[int64][]FixtureCandidate // keyed by league_id
}

func (l *fakeLookup) ListFixturesForLeague(ctx context.Context, leagueID int64, start, end time.Time) ([]FixtureCandidate, error) {
	return l.candidates[leagueID], nil
}

type fakeStore struct {
	written []ParsedOdds
}

func (s *fakeStore) UpsertOdds(ctx context.Context, parsed []ParsedOdds) (int, error) {
	s.written = append(s.written, parsed...)
	return len(parsed), nil
}

func TestService_RunCycle_MatchesAndWritesOdds(t *testing.T) {
	kickoff := time.Now().UTC().Add(24 * time.Hour)

	fetcher := &fakeFetcher{
		events: map[string][]Event{
			"soccer_epl": {{
				HomeTeam:     "Arsenal",
				AwayTeam:     "Chelsea",
				CommenceTime: kickoff,
				Bookmakers: []Bookmaker{{
					Key: "bookie-a",
					Markets: []Market{{
						Key: "h2h",
						Outcomes: []Outcome{
							{Name: "Arsenal", Price: 2.0},
							{Name: "Draw", Price: 3.0},
							{Name: "Chelsea", Price: 4.0},
						},
					}},
				}},
			}},
		},
	}
	lookup := &fakeLookup{
		candidates: map[int64][]FixtureCandidate{
			8: {{FixtureID: 1, HomeName: "Arsenal", AwayName: "Chelsea", StartingAt: kickoff}},
		},
	}
	store := &fakeStore{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, lookup, store, 3, logger)
	svc.runCycle(context.Background())

	if len(store.written) != 1 {
		t.Fatalf("got %d written odds rows, want 1", len(store.written))
	}
	if store.written[0].FixtureID != 1 {
		t.Fatalf("got fixture_id %d, want 1", store.written[0].FixtureID)
	}
}

func TestService_RunCycle_MergesPerEventBTTSOdds(t *testing.T) {
	kickoff := time.Now().UTC().Add(24 * time.Hour)

	fetcher := &fakeFetcher{
		events: map[string][]Event{
			"soccer_epl": {{
				ID:           "evt-1",
				HomeTeam:     "Arsenal",
				AwayTeam:     "Chelsea",
				CommenceTime: kickoff,
				Bookmakers: []Bookmaker{{
					Key: "bookie-a",
					Markets: []Market{{
						Key: "h2h",
						Outcomes: []Outcome{
							{Name: "Arsenal", Price: 2.0},
							{Name: "Draw", Price: 3.0},
							{Name: "Chelsea", Price: 4.0},
						},
					}},
				}},
			}},
		},
		bttsEvents: map[string]Event{
			"evt-1": {
				ID: "evt-1",
				Bookmakers: []Bookmaker{{
					Key: "bookie-a",
					Markets: []Market{{
						Key: "btts",
						Outcomes: []Outcome{
							{Name: "Yes", Price: 1.8},
							{Name: "No", Price: 2.0},
						},
					}},
				}},
			},
		},
	}
	lookup := &fakeLookup{
		candidates: map[int64][]FixtureCandidate{
			8: {{FixtureID: 1, HomeName: "Arsenal", AwayName: "Chelsea", StartingAt: kickoff}},
		},
	}
	store := &fakeStore{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, lookup, store, 3, logger)
	svc.runCycle(context.Background())

	if len(store.written) != 1 {
		t.Fatalf("got %d written odds rows, want 1", len(store.written))
	}
	if !store.written[0].HasBTTS {
		t.Fatal("expected HasBTTS true after merging the per-event btts fetch")
	}
}

func TestService_RunCycle_SkipsFetchWhenNoCandidatesInLeague(t *testing.T) {
	fetcher := &fakeFetcher{events: map[string][]Event{}}
	lookup := &fakeLookup{candidates: map[int64][]FixtureCandidate{}} // no leagues have candidates
	store := &fakeStore{}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))

	svc := NewService(fetcher, lookup, store, 3, logger)
	svc.runCycle(context.Background())

	if fetcher.calls != 0 {
		t.Fatalf("got %d fetch calls, want 0 when no league has candidates", fetcher.calls)
	}
	if len(store.written) != 0 {
		t.Fatalf("got %d written odds rows, want 0", len(store.written))
	}
}
