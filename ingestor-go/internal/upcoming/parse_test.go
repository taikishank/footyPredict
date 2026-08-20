package upcoming

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

func TestParseFixture(t *testing.T) {
	f := mustFixture(t, `{
		"id": 1, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"state": {"short_name": "NS"},
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}},
			{"id": 20, "name": "Away FC", "meta": {"location": "away"}}
		]
	}`)

	pf, ok, err := parseFixture(f)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ok {
		t.Fatal("expected fixture to parse")
	}
	if pf.FixtureID != 1 || pf.HomeID != 10 || pf.AwayID != 20 {
		t.Fatalf("got %+v, want fixture 1 home=10 away=20", pf)
	}
	if pf.HomeName != "Home FC" || pf.AwayName != "Away FC" {
		t.Fatalf("got %+v, want home/away names Home FC / Away FC", pf)
	}
}

func TestParseFixture_MissingParticipantErrors(t *testing.T) {
	f := mustFixture(t, `{
		"id": 2, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
		"state": {"short_name": "NS"},
		"participants": [
			{"id": 10, "name": "Home FC", "meta": {"location": "home"}}
		]
	}`)

	_, _, err := parseFixture(f)
	if err == nil {
		t.Fatal("expected error for missing away participant")
	}
}
