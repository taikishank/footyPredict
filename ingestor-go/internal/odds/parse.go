package odds

import "fmt"

// ParsedOdds holds one fixture's averaged, de-vigged h2h prices, ready to
// upsert into the odds table.
type ParsedOdds struct {
	FixtureID      int64
	BookmakerCount int
	HomePrice      float64
	DrawPrice      float64
	AwayPrice      float64
	ImpliedHome    float64
	ImpliedDraw    float64
	ImpliedAway    float64
}

// parseEvent averages an Event's h2h decimal prices across every bookmaker
// that quoted all three outcomes, then de-vigs the average via proportional
// normalization (each implied probability divided by the sum of all three,
// so they sum to 1). Returns (zero, false) if no bookmaker quoted a
// complete home/draw/away h2h market.
func parseEvent(fixtureID int64, event Event) (ParsedOdds, bool) {
	var sumHome, sumDraw, sumAway float64
	count := 0

	for _, bm := range event.Bookmakers {
		home, draw, away, ok := h2hPrices(bm, event.HomeTeam, event.AwayTeam)
		if !ok {
			continue
		}
		sumHome += home
		sumDraw += draw
		sumAway += away
		count++
	}
	if count == 0 {
		return ParsedOdds{}, false
	}

	avgHome := sumHome / float64(count)
	avgDraw := sumDraw / float64(count)
	avgAway := sumAway / float64(count)

	rawHome, rawDraw, rawAway := 1/avgHome, 1/avgDraw, 1/avgAway
	overround := rawHome + rawDraw + rawAway

	return ParsedOdds{
		FixtureID:      fixtureID,
		BookmakerCount: count,
		HomePrice:      avgHome,
		DrawPrice:      avgDraw,
		AwayPrice:      avgAway,
		ImpliedHome:    rawHome / overround,
		ImpliedDraw:    rawDraw / overround,
		ImpliedAway:    rawAway / overround,
	}, true
}

// h2hPrices extracts (home, draw, away) decimal prices from a bookmaker's
// h2h market. Outcome.Name is the literal team name for home/away and
// "Draw" for the draw - matched against the event's own team names rather
// than positionally, since outcome order isn't guaranteed.
func h2hPrices(bm Bookmaker, homeTeam, awayTeam string) (home, draw, away float64, ok bool) {
	for _, mkt := range bm.Markets {
		if mkt.Key != "h2h" {
			continue
		}
		var haveHome, haveDraw, haveAway bool
		for _, o := range mkt.Outcomes {
			switch o.Name {
			case homeTeam:
				home, haveHome = o.Price, true
			case awayTeam:
				away, haveAway = o.Price, true
			case "Draw":
				draw, haveDraw = o.Price, true
			}
		}
		if haveHome && haveDraw && haveAway {
			return home, draw, away, true
		}
	}
	return 0, 0, 0, false
}

func (p ParsedOdds) String() string {
	return fmt.Sprintf("fixture=%d home=%.2f draw=%.2f away=%.2f (n=%d bookmakers)",
		p.FixtureID, p.ImpliedHome, p.ImpliedDraw, p.ImpliedAway, p.BookmakerCount)
}
