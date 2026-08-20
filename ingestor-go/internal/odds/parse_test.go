package odds

import "testing"

func TestParseEvent_AveragesAndDeVigs(t *testing.T) {
	event := Event{
		HomeTeam: "Arsenal",
		AwayTeam: "Chelsea",
		Bookmakers: []Bookmaker{
			{
				Key: "bookie-a",
				Markets: []Market{{
					Key: "h2h",
					Outcomes: []Outcome{
						{Name: "Arsenal", Price: 2.0},
						{Name: "Draw", Price: 3.0},
						{Name: "Chelsea", Price: 4.0},
					},
				}},
			},
			{
				Key: "bookie-b",
				Markets: []Market{{
					Key: "h2h",
					Outcomes: []Outcome{
						{Name: "Arsenal", Price: 2.2},
						{Name: "Draw", Price: 3.2},
						{Name: "Chelsea", Price: 3.6},
					},
				}},
			},
		},
	}

	got, ok := parseEvent(1, event)
	if !ok {
		t.Fatal("expected a parsed result")
	}
	if got.BookmakerCount != 2 {
		t.Fatalf("got bookmaker count %d, want 2", got.BookmakerCount)
	}

	sum := got.ImpliedHome + got.ImpliedDraw + got.ImpliedAway
	if diff := sum - 1.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("implied probabilities sum to %v, want 1.0", sum)
	}
	if got.ImpliedHome <= got.ImpliedDraw || got.ImpliedHome <= got.ImpliedAway {
		t.Fatalf("expected home (shortest average price) to have the highest implied probability, got %+v", got)
	}
}

func TestParseEvent_SkipsBookmakerMissingAnOutcome(t *testing.T) {
	event := Event{
		HomeTeam: "Arsenal",
		AwayTeam: "Chelsea",
		Bookmakers: []Bookmaker{
			{
				Key: "incomplete",
				Markets: []Market{{
					Key: "h2h",
					Outcomes: []Outcome{
						{Name: "Arsenal", Price: 2.0},
						{Name: "Chelsea", Price: 4.0},
						// no Draw outcome
					},
				}},
			},
		},
	}

	_, ok := parseEvent(1, event)
	if ok {
		t.Fatal("expected no result when no bookmaker has a complete h2h market")
	}
}

func TestParseEvent_NoBookmakersNoResult(t *testing.T) {
	event := Event{HomeTeam: "Arsenal", AwayTeam: "Chelsea"}

	_, ok := parseEvent(1, event)
	if ok {
		t.Fatal("expected no result for an event with no bookmakers")
	}
}
