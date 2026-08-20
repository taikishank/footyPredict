package upcoming

import (
	"fmt"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

// ParsedFixture holds the fields needed to seed a fixtures row ahead of
// kickoff - no result yet, unlike ingest.ParsedFixture.
type ParsedFixture struct {
	FixtureID  int64
	LeagueID   int64
	StartingAt time.Time
	HomeID     int64
	AwayID     int64
	HomeName   string
	AwayName   string
	Raw        []byte
}

// parseFixture extracts the fields the upcoming-fixtures pipeline needs to
// persist. It returns (zero, false) for fixtures missing a home/away
// participant.
func parseFixture(f sportmonks.Fixture) (ParsedFixture, bool, error) {
	home, homeOK := f.Participant("home")
	away, awayOK := f.Participant("away")
	if !homeOK || !awayOK {
		return ParsedFixture{}, false, fmt.Errorf("fixture %d missing home/away participant", f.ID)
	}

	startingAt, err := time.Parse("2006-01-02 15:04:05", f.StartingAt)
	if err != nil {
		return ParsedFixture{}, false, fmt.Errorf("fixture %d: parsing starting_at %q: %w", f.ID, f.StartingAt, err)
	}

	return ParsedFixture{
		FixtureID:  f.ID,
		LeagueID:   f.LeagueID,
		StartingAt: startingAt,
		HomeID:     home.ID,
		AwayID:     away.ID,
		HomeName:   home.Name,
		AwayName:   away.Name,
		Raw:        f.Raw,
	}, true, nil
}
