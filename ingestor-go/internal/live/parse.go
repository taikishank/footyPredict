// Package live polls SportMonks for in-play fixtures and fans them out to a
// match-state store and an event publisher (Phase 3's local stand-ins for
// DynamoDB and SQS - see Service).
package live

import (
	"fmt"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

const currentScoreTypeID = 1525

// MatchState is a snapshot of one fixture's live score, mirroring what will
// eventually live in DynamoDB.
type MatchState struct {
	FixtureID int64
	State     string // SportMonks state short_name, e.g. "1H", "2H", "HT"
	HomeGoals int
	AwayGoals int
}

// Event is a single in-match event, mirroring what will eventually be
// published to SQS.
type Event struct {
	FixtureID     int64
	EventID       int64
	TypeID        int64
	ParticipantID int64
	PlayerName    string
	Minute        int
}

// parseLive extracts the live match state and any events from a fixture.
// Unlike the batch pipeline's parseFixture, it doesn't require a final score
// - an in-play fixture may not have one yet - so it returns ok=false only
// when the fixture can't be identified as belonging to a specific match.
func parseLive(f sportmonks.Fixture) (MatchState, []Event, bool, error) {
	if f.ID == 0 {
		return MatchState{}, nil, false, fmt.Errorf("fixture missing id")
	}

	homeGoals, awayGoals := 0, 0
	for _, s := range f.Scores {
		if s.TypeID != currentScoreTypeID {
			continue
		}
		switch s.Score.Participant {
		case "home":
			homeGoals = s.Score.Goals
		case "away":
			awayGoals = s.Score.Goals
		}
	}

	state := MatchState{
		FixtureID: f.ID,
		State:     f.State.ShortName,
		HomeGoals: homeGoals,
		AwayGoals: awayGoals,
	}

	events := make([]Event, 0, len(f.Events))
	for _, e := range f.Events {
		events = append(events, Event{
			FixtureID:     f.ID,
			EventID:       e.ID,
			TypeID:        e.TypeID,
			ParticipantID: e.ParticipantID,
			PlayerName:    e.PlayerName,
			Minute:        e.Minute,
		})
	}

	return state, events, true, nil
}
