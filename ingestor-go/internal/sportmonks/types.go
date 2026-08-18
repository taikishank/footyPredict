package sportmonks

import "encoding/json"

// finishedStates are the SportMonks fixture state short_names that indicate
// a match has a final result. Postponed/cancelled/abandoned fixtures are
// deliberately excluded - there is no result to ingest.
var finishedStates = map[string]bool{
	"FT":      true,
	"AET":     true,
	"FT_PEN":  true,
	"AWARDED": true,
}

type Fixture struct {
	ID           int64           `json:"id"`
	LeagueID     int64           `json:"league_id"`
	StartingAt   string          `json:"starting_at"`
	State        State           `json:"state"`
	Participants []Participant   `json:"participants"`
	Scores       []Score         `json:"scores"`
	Statistics   []Statistic     `json:"statistics"`
	Events       []Event         `json:"events"`
	Raw          json.RawMessage `json:"-"`
}

// Event is a single in-match event (goal, card, substitution, ...) as
// returned by the events include.
type Event struct {
	ID            int64  `json:"id"`
	TypeID        int64  `json:"type_id"`
	ParticipantID int64  `json:"participant_id"`
	PlayerName    string `json:"player_name"`
	Minute        int    `json:"minute"`
}

type State struct {
	ShortName string `json:"short_name"`
}

func (f Fixture) IsFinished() bool {
	return finishedStates[f.State.ShortName]
}

type Participant struct {
	ID   int64  `json:"id"`
	Name string `json:"name"`
	Meta struct {
		Location string `json:"location"` // "home" or "away"
	} `json:"meta"`
}

type Score struct {
	TypeID int64 `json:"type_id"`
	Score  struct {
		Participant string `json:"participant"` // "home" or "away"
		Goals       int    `json:"goals"`
	} `json:"score"`
}

type Statistic struct {
	TypeID        int64 `json:"type_id"`
	ParticipantID int64 `json:"participant_id"`
	Data          struct {
		Value *float64 `json:"value"`
	} `json:"data"`
}

type fixturesResponse struct {
	Data      []json.RawMessage `json:"data"`
	RateLimit *RateLimit        `json:"rate_limit"`
}

// RateLimit mirrors the rate_limit object SportMonks includes on every v3
// response, reflecting the account's actual plan quota for the current hour.
type RateLimit struct {
	Remaining       int `json:"remaining"`
	ResetsInSeconds int `json:"resets_in_seconds"`
}
