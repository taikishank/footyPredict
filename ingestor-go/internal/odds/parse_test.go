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

func TestParseEvent_MergesBTTSAlongsideH2H(t *testing.T) {
	event := Event{
		HomeTeam: "Arsenal",
		AwayTeam: "Chelsea",
		Bookmakers: []Bookmaker{
			{
				Key: "bookie-a",
				Markets: []Market{
					{
						Key: "h2h",
						Outcomes: []Outcome{
							{Name: "Arsenal", Price: 2.0},
							{Name: "Draw", Price: 3.0},
							{Name: "Chelsea", Price: 4.0},
						},
					},
					{
						Key: "btts",
						Outcomes: []Outcome{
							{Name: "Yes", Price: 1.8},
							{Name: "No", Price: 2.0},
						},
					},
				},
			},
		},
	}

	got, ok := parseEvent(1, event)
	if !ok {
		t.Fatal("expected a parsed result")
	}
	if !got.HasBTTS {
		t.Fatal("expected HasBTTS to be true")
	}
	if got.BTTSBookmakerCount != 1 {
		t.Fatalf("got btts bookmaker count %d, want 1", got.BTTSBookmakerCount)
	}
	sum := got.ImpliedBTTSYes + got.ImpliedBTTSNo
	if diff := sum - 1.0; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("implied btts probabilities sum to %v, want 1.0", sum)
	}
	if got.ImpliedBTTSYes <= got.ImpliedBTTSNo {
		t.Fatalf("expected yes (shorter price) to have the higher implied probability, got %+v", got)
	}
}

func TestParseEvent_H2HWithoutBTTSLeavesHasBTTSFalse(t *testing.T) {
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
		},
	}

	got, ok := parseEvent(1, event)
	if !ok {
		t.Fatal("expected a parsed result")
	}
	if got.HasBTTS {
		t.Fatalf("expected HasBTTS false with no btts market quoted, got %+v", got)
	}
}

func TestBTTSPrices_RequiresBothOutcomes(t *testing.T) {
	bm := Bookmaker{
		Markets: []Market{{
			Key:      "btts",
			Outcomes: []Outcome{{Name: "Yes", Price: 1.8}},
		}},
	}
	if _, _, ok := bttsPrices(bm); ok {
		t.Fatal("expected no result when the No outcome is missing")
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
