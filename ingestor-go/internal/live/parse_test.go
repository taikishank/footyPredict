package live

import (
	"encoding/json"
	"testing"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

func TestParseLive_ExtractsStateScoreAndEvents(t *testing.T) {
	raw := `{
		"id": 42, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"state": {"short_name": "2H"},
		"scores": [
			{"type_id": 1525, "score": {"participant": "home", "goals": 2}},
			{"type_id": 1525, "score": {"participant": "away", "goals": 1}}
		],
		"events": [
			{"id": 100, "type_id": 14, "participant_id": 10, "player_name": "Player A", "minute": 55}
		]
	}`
	var f sportmonks.Fixture
	if err := json.Unmarshal([]byte(raw), &f); err != nil {
		t.Fatalf("unmarshaling fixture: %v", err)
	}

	state, events, ok, err := parseLive(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true")
	}
	if state.FixtureID != 42 || state.State != "2H" || state.HomeGoals != 2 || state.AwayGoals != 1 {
		t.Fatalf("got state %+v", state)
	}
	if len(events) != 1 || events[0].EventID != 100 || events[0].Minute != 55 {
		t.Fatalf("got events %+v", events)
	}
}

func TestParseLive_ErrorsWithoutFixtureID(t *testing.T) {
	_, _, ok, err := parseLive(sportmonks.Fixture{})
	if err == nil || ok {
		t.Fatalf("expected error and ok=false, got ok=%v err=%v", ok, err)
	}
}
