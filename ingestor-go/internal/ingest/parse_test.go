package ingest

import (
	"encoding/json"
	"testing"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

func mustFixture(t *testing.T, raw string) sportmonks.Fixture {
	t.Helper()
	var f sportmonks.Fixture
	if err := json.Unmarshal([]byte(raw), &f); err != nil {
		t.Fatalf("unmarshaling fixture: %v", err)
	}
	f.Raw = json.RawMessage(raw)
	return f
}

func TestParseFixture_FinalScore(t *testing.T) {
	f := mustFixture(t, `{
		"id": 1, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}},
			{"id": 20, "name": "Away FC", "meta": {"location": "away"}}
		],
		"scores": [
			{"type_id": 1525, "score": {"participant": "home", "goals": 2}},
			{"type_id": 1525, "score": {"participant": "away", "goals": 1}}
		],
		"statistics": []
	}`)

	pf, ok, err := parseFixture(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected fixture to parse")
	}
	if pf.HomeGoals != 2 || pf.AwayGoals != 1 {
		t.Fatalf("got goals %d-%d, want 2-1", pf.HomeGoals, pf.AwayGoals)
	}
	if pf.Result != "home_win" {
		t.Fatalf("got result %q, want home_win", pf.Result)
	}
}

func TestParseFixture_FallsBackToGoalsStat(t *testing.T) {
	f := mustFixture(t, `{
		"id": 2, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}},
			{"id": 20, "name": "Away FC", "meta": {"location": "away"}}
		],
		"scores": [],
		"statistics": [
			{"type_id": 52, "participant_id": 10, "data": {"value": 3}},
			{"type_id": 52, "participant_id": 20, "data": {"value": 3}}
		]
	}`)

	pf, ok, err := parseFixture(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected fixture to parse via fallback")
	}
	if pf.Result != "draw" {
		t.Fatalf("got result %q, want draw", pf.Result)
	}
}

func TestParseFixture_NoResultSkipped(t *testing.T) {
	f := mustFixture(t, `{
		"id": 3, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}},
			{"id": 20, "name": "Away FC", "meta": {"location": "away"}}
		],
		"scores": [],
		"statistics": []
	}`)

	_, ok, err := parseFixture(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ok {
		t.Fatal("expected fixture with no derivable score to be skipped")
	}
}

func TestParseFixture_MissingParticipantErrors(t *testing.T) {
	f := mustFixture(t, `{
		"id": 4, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}}
		],
		"scores": [],
		"statistics": []
	}`)

	_, _, err := parseFixture(f)
	if err == nil {
		t.Fatal("expected error for missing away participant")
	}
}
