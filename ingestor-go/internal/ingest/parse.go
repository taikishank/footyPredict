package ingest

import (
	"fmt"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
)

// finalScoreTypeID / goalsStatTypeID mirror the constants used by
// ml/build_features.py so the two pipelines agree on how a fixture's result
// is derived from the SportMonks payload.
const (
	finalScoreTypeID = 1525
	goalsStatTypeID  = 52
)

type ParsedFixture struct {
	FixtureID  int64
	LeagueID   int64
	StartingAt time.Time
	HomeID     int64
	AwayID     int64
	HomeName   string
	AwayName   string
	HomeGoals  int
	AwayGoals  int
	Result     string // "home_win" | "away_win" | "draw"
	Raw        []byte
}

// parseFixture extracts the fields the batch pipeline needs to persist. It
// returns (zero, false) for fixtures with no derivable final score, mirroring
// build_features.py's parse_fixture behavior of skipping those.
func parseFixture(f sportmonks.Fixture) (ParsedFixture, bool, error) {
	home, homeOK := f.Participant("home")
	away, awayOK := f.Participant("away")
	if !homeOK || !awayOK {
		return ParsedFixture{}, false, fmt.Errorf("fixture %d missing home/away participant", f.ID)
	}

	homeGoals, awayGoals, ok := finalGoals(f, home.ID, away.ID)
	if !ok {
		return ParsedFixture{}, false, nil
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
		HomeGoals:  homeGoals,
		AwayGoals:  awayGoals,
		Result:     result(homeGoals, awayGoals),
		Raw:        f.Raw,
	}, true, nil
}

// finalGoals prefers the CURRENT score entry (type_id 1525); if a fixture
// doesn't carry one it falls back to the goals statistic, same fallback
// build_features.py uses.
func finalGoals(f sportmonks.Fixture, homeID, awayID int64) (home, away int, ok bool) {
	haveHome, haveAway := false, false
	for _, s := range f.Scores {
		if s.TypeID != finalScoreTypeID {
			continue
		}
		switch s.Score.Participant {
		case "home":
			home, haveHome = s.Score.Goals, true
		case "away":
			away, haveAway = s.Score.Goals, true
		}
	}
	if haveHome && haveAway {
		return home, away, true
	}

	for _, s := range f.Statistics {
		if s.TypeID != goalsStatTypeID || s.Data.Value == nil {
			continue
		}
		switch s.ParticipantID {
		case homeID:
			home, haveHome = int(*s.Data.Value), true
		case awayID:
			away, haveAway = int(*s.Data.Value), true
		}
	}
	return home, away, haveHome && haveAway
}

func result(homeGoals, awayGoals int) string {
	switch {
	case homeGoals > awayGoals:
		return "home_win"
	case awayGoals > homeGoals:
		return "away_win"
	default:
		return "draw"
	}
}
