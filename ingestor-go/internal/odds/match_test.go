package odds

import (
	"testing"
	"time"
)

func TestMatchFixture_ExactNamesWithinTolerance(t *testing.T) {
	kickoff := time.Date(2026, 1, 10, 15, 0, 0, 0, time.UTC)
	candidates := []FixtureCandidate{
		{FixtureID: 1, HomeName: "Arsenal", AwayName: "Chelsea", StartingAt: kickoff},
	}
	event := Event{HomeTeam: "Arsenal", AwayTeam: "Chelsea", CommenceTime: kickoff.Add(10 * time.Minute)}

	id, ok := matchFixture(event, candidates)
	if !ok || id != 1 {
		t.Fatalf("got (%d, %v), want (1, true)", id, ok)
	}
}

func TestMatchFixture_AbbreviatedNameStillMatches(t *testing.T) {
	kickoff := time.Date(2026, 1, 10, 15, 0, 0, 0, time.UTC)
	candidates := []FixtureCandidate{
		{FixtureID: 1, HomeName: "Manchester United", AwayName: "Manchester City", StartingAt: kickoff},
	}
	event := Event{HomeTeam: "Manchester United", AwayTeam: "Man City", CommenceTime: kickoff}

	id, ok := matchFixture(event, candidates)
	if !ok || id != 1 {
		t.Fatalf("got (%d, %v), want (1, true)", id, ok)
	}
}

func TestMatchFixture_OutsideToleranceNoMatch(t *testing.T) {
	kickoff := time.Date(2026, 1, 10, 15, 0, 0, 0, time.UTC)
	candidates := []FixtureCandidate{
		{FixtureID: 1, HomeName: "Arsenal", AwayName: "Chelsea", StartingAt: kickoff},
	}
	event := Event{HomeTeam: "Arsenal", AwayTeam: "Chelsea", CommenceTime: kickoff.Add(3 * time.Hour)}

	_, ok := matchFixture(event, candidates)
	if ok {
		t.Fatal("expected no match outside kickoff tolerance")
	}
}

func TestMatchFixture_DifferentTeamsNoMatch(t *testing.T) {
	kickoff := time.Date(2026, 1, 10, 15, 0, 0, 0, time.UTC)
	candidates := []FixtureCandidate{
		{FixtureID: 1, HomeName: "Arsenal", AwayName: "Chelsea", StartingAt: kickoff},
	}
	event := Event{HomeTeam: "Liverpool", AwayTeam: "Everton", CommenceTime: kickoff}

	_, ok := matchFixture(event, candidates)
	if ok {
		t.Fatal("expected no match for unrelated teams")
	}
}

func TestMatchFixture_PicksBestOfMultipleCandidates(t *testing.T) {
	kickoff := time.Date(2026, 1, 10, 15, 0, 0, 0, time.UTC)
	candidates := []FixtureCandidate{
		{FixtureID: 1, HomeName: "Manchester United", AwayName: "Arsenal", StartingAt: kickoff},
		{FixtureID: 2, HomeName: "Manchester City", AwayName: "Chelsea", StartingAt: kickoff},
	}
	event := Event{HomeTeam: "Manchester City", AwayTeam: "Chelsea", CommenceTime: kickoff}

	id, ok := matchFixture(event, candidates)
	if !ok || id != 2 {
		t.Fatalf("got (%d, %v), want (2, true)", id, ok)
	}
}
