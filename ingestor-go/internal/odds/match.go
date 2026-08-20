package odds

import (
	"regexp"
	"strings"
	"time"
)

// kickoffTolerance is how far an odds Event's commence_time may drift from a
// fixture's starting_at and still be considered the same match. SportMonks
// and The Odds API occasionally disagree on kickoff time by a few minutes;
// 90 minutes comfortably covers that without risking a same-day
// doubleheader collision for two different fixtures.
const kickoffTolerance = 90 * time.Minute

// minNameScore is the minimum Jaccard token-overlap score (see nameScore)
// both the home and away team names must clear for an Event to be
// considered a match. Team names between the two APIs aren't always
// identical ("Man United" vs "Manchester United"), so this is a heuristic,
// not an exact match - unmatched events are simply skipped and logged.
const minNameScore = 0.5

var nonAlnum = regexp.MustCompile(`[^a-z0-9\s]`)

// clubStopWords are generic club-name tokens stripped before comparing, so
// they don't inflate the overlap score between two otherwise-different clubs.
var clubStopWords = map[string]bool{
	"fc": true, "afc": true, "cf": true, "cd": true, "sc": true,
	"ac": true, "club": true,
}

// tokenAliases maps common abbreviations odds providers use to the token a
// SportMonks-style full name would produce, so e.g. "Man City" and
// "Manchester City" share a token instead of scoring as unrelated.
var tokenAliases = map[string]string{
	"man":    "manchester",
	"utd":    "united",
	"spurs":  "tottenham",
	"wolves": "wolverhampton",
}

// normalizeTeamName lowercases, strips punctuation, removes generic club
// stopwords, and applies tokenAliases, returning the token set used by
// nameScore.
func normalizeTeamName(name string) map[string]bool {
	lower := strings.ToLower(name)
	stripped := nonAlnum.ReplaceAllString(lower, " ")
	tokens := map[string]bool{}
	for _, tok := range strings.Fields(stripped) {
		if clubStopWords[tok] {
			continue
		}
		if alias, ok := tokenAliases[tok]; ok {
			tok = alias
		}
		tokens[tok] = true
	}
	return tokens
}

// nameScore is the Jaccard similarity of two team names' token sets.
func nameScore(a, b string) float64 {
	ta, tb := normalizeTeamName(a), normalizeTeamName(b)
	if len(ta) == 0 || len(tb) == 0 {
		return 0
	}
	intersection := 0
	for tok := range ta {
		if tb[tok] {
			intersection++
		}
	}
	union := len(ta) + len(tb) - intersection
	return float64(intersection) / float64(union)
}

// matchFixture finds the candidate that best matches an odds Event, requiring
// both team names to clear minNameScore and the kickoff time to fall within
// kickoffTolerance. Returns (0, false) if no candidate qualifies.
func matchFixture(event Event, candidates []FixtureCandidate) (int64, bool) {
	var best FixtureCandidate
	bestScore := -1.0

	for _, c := range candidates {
		if diff := event.CommenceTime.Sub(c.StartingAt); diff > kickoffTolerance || diff < -kickoffTolerance {
			continue
		}
		homeScore := nameScore(event.HomeTeam, c.HomeName)
		awayScore := nameScore(event.AwayTeam, c.AwayName)
		if homeScore < minNameScore || awayScore < minNameScore {
			continue
		}
		combined := homeScore + awayScore
		if combined > bestScore {
			bestScore = combined
			best = c
		}
	}

	if bestScore < 0 {
		return 0, false
	}
	return best.FixtureID, true
}
