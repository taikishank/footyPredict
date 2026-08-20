// Package odds pulls bookmaker odds from The Odds API for fixtures already
// seeded by the upcoming-fixtures pipeline, matches each odds event to a
// fixture, de-vigs the prices into implied probabilities, and upserts the
// result into Postgres (PROJECT_SPEC.md Phase 4).
package odds

import "time"

// leagueSportKeys maps a SportMonks league_id (see
// sm_league_id_to_raw_name_mapping.txt) to The Odds API's sport_key for that
// league. Only the five leagues the rest of the pipeline tracks are listed;
// fixtures in other leagues are never queried for odds.
var leagueSportKeys = map[int64]string{
	8:   "soccer_epl",
	82:  "soccer_germany_bundesliga",
	301: "soccer_france_ligue_one",
	384: "soccer_italy_serie_a",
	564: "soccer_spain_la_liga",
}

// Event is a single fixture's odds as returned by The Odds API's
// /v4/sports/{sport}/odds endpoint (regions=eu, markets=h2h).
type Event struct {
	ID           string      `json:"id"`
	SportKey     string      `json:"sport_key"`
	CommenceTime time.Time   `json:"commence_time"`
	HomeTeam     string      `json:"home_team"`
	AwayTeam     string      `json:"away_team"`
	Bookmakers   []Bookmaker `json:"bookmakers"`
}

type Bookmaker struct {
	Key     string   `json:"key"`
	Markets []Market `json:"markets"`
}

type Market struct {
	Key      string    `json:"key"`
	Outcomes []Outcome `json:"outcomes"`
}

// Outcome.Name is either a team name (matching Event.HomeTeam/AwayTeam
// verbatim) or "Draw" for the h2h market.
type Outcome struct {
	Name  string  `json:"name"`
	Price float64 `json:"price"`
}

// FixtureCandidate is a fixture already in Postgres (seeded by the upcoming
// poller) that an odds Event could match against.
type FixtureCandidate struct {
	FixtureID  int64
	HomeName   string
	AwayName   string
	StartingAt time.Time
}
