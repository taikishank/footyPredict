package odds

import "fmt"

// ParsedOdds holds one fixture's averaged, de-vigged h2h prices, ready to
// upsert into the odds table. The btts fields are a second, independent
// market on the same fixture - HasBTTS is false (and the fields zero) when
// no bookmaker quoted a complete yes/no btts market, which happens on its
// own schedule from h2h: a fixture can have one market without the other.
type ParsedOdds struct {
	FixtureID      int64
	BookmakerCount int
	HomePrice      float64
	DrawPrice      float64
	AwayPrice      float64
	ImpliedHome    float64
	ImpliedDraw    float64
	ImpliedAway    float64

	HasBTTS            bool
	BTTSBookmakerCount int
	BTTSYesPrice       float64
	BTTSNoPrice        float64
	ImpliedBTTSYes     float64
	ImpliedBTTSNo      float64
}

// parseEvent averages an Event's h2h decimal prices across every bookmaker
// that quoted all three outcomes, then de-vigs the average via proportional
// normalization (each implied probability divided by the sum of all three,
// so they sum to 1). Returns (zero, false) if no bookmaker quoted a
// complete home/draw/away h2h market - h2h is the market every downstream
// consumer requires, so its absence still fails the whole event. btts is
// parsed and merged in regardless, via its own ok flag on HasBTTS.
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

	parsed := ParsedOdds{
		FixtureID:      fixtureID,
		BookmakerCount: count,
		HomePrice:      avgHome,
		DrawPrice:      avgDraw,
		AwayPrice:      avgAway,
		ImpliedHome:    rawHome / overround,
		ImpliedDraw:    rawDraw / overround,
		ImpliedAway:    rawAway / overround,
	}

	if yes, no, btttsCount, ok := averageBTTSPrices(event); ok {
		rawYes, rawNo := 1/yes, 1/no
		bttsOverround := rawYes + rawNo
		parsed.HasBTTS = true
		parsed.BTTSBookmakerCount = btttsCount
		parsed.BTTSYesPrice = yes
		parsed.BTTSNoPrice = no
		parsed.ImpliedBTTSYes = rawYes / bttsOverround
		parsed.ImpliedBTTSNo = rawNo / bttsOverround
	}

	return parsed, true
}

// averageBTTSPrices mirrors parseEvent's h2h averaging loop, for the
// two-outcome btts market instead of the three-outcome h2h one.
func averageBTTSPrices(event Event) (yes, no float64, count int, ok bool) {
	var sumYes, sumNo float64
	for _, bm := range event.Bookmakers {
		y, n, ok := bttsPrices(bm)
		if !ok {
			continue
		}
		sumYes += y
		sumNo += n
		count++
	}
	if count == 0 {
		return 0, 0, 0, false
	}
	return sumYes / float64(count), sumNo / float64(count), count, true
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

// bttsPrices extracts (yes, no) decimal prices from a bookmaker's btts
// market. h2hPrices's twin, for a two-outcome market matched by name
// instead of by team.
func bttsPrices(bm Bookmaker) (yes, no float64, ok bool) {
	for _, mkt := range bm.Markets {
		if mkt.Key != "btts" {
			continue
		}
		var haveYes, haveNo bool
		for _, o := range mkt.Outcomes {
			switch o.Name {
			case "Yes":
				yes, haveYes = o.Price, true
			case "No":
				no, haveNo = o.Price, true
			}
		}
		if haveYes && haveNo {
			return yes, no, true
		}
	}
	return 0, 0, false
}

func (p ParsedOdds) String() string {
	btts := "no btts"
	if p.HasBTTS {
		btts = fmt.Sprintf("btts yes=%.2f no=%.2f (n=%d)",
			p.ImpliedBTTSYes, p.ImpliedBTTSNo, p.BTTSBookmakerCount)
	}
	return fmt.Sprintf("fixture=%d home=%.2f draw=%.2f away=%.2f (n=%d bookmakers) %s",
		p.FixtureID, p.ImpliedHome, p.ImpliedDraw, p.ImpliedAway, p.BookmakerCount, btts)
}
