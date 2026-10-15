package odds

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

const defaultBaseURL = "https://api.the-odds-api.com/v4"

type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// FetchOdds returns every upcoming event The Odds API has h2h prices for in
// the given sport, from eu bookmakers.
//
// btts used to ride along on this same call (markets=h2h,btts) on the
// strength of the API accepting a comma-separated markets list, but the
// bulk /sports/{sport}/odds endpoint rejects btts outright - 422
// INVALID_MARKET - because it's one of The Odds API's "additional markets",
// only served per-event via FetchEventBTTS below. The combined call was
// failing every poll, h2h included, until this was split back out.
func (c *Client) FetchOdds(ctx context.Context, sportKey string) ([]Event, error) {
	url := fmt.Sprintf("%s/sports/%s/odds", c.baseURL, sportKey)
	q := urlValues{"regions": "eu", "markets": "h2h", "oddsFormat": "decimal"}

	var events []Event
	if err := c.getJSON(ctx, url, q, &events); err != nil {
		return nil, err
	}
	return events, nil
}

// FetchEventBTTS returns one event's btts bookmaker quotes, via The Odds
// API's per-event endpoint - the only place it serves "additional markets"
// like btts, unlike h2h which the bulk endpoint covers for every event in
// one call. This costs its own request (and its own metered credit) per
// event, not per league, so the service only calls it for events FetchOdds
// already matched to a tracked fixture - never for every candidate.
//
// The endpoint's response is shaped exactly like one element of FetchOdds's
// array (same id/sport_key/commence_time/home_team/away_team/bookmakers
// fields), so it decodes straight into Event. Bookmakers is empty, not an
// error, when nobody has quoted btts for this event yet - btts opens on its
// own schedule relative to h2h (see parse.go's ParsedOdds doc).
func (c *Client) FetchEventBTTS(ctx context.Context, sportKey, eventID string) (Event, error) {
	url := fmt.Sprintf("%s/sports/%s/events/%s/odds", c.baseURL, sportKey, eventID)
	q := urlValues{"regions": "eu", "markets": "btts", "oddsFormat": "decimal"}

	var event Event
	if err := c.getJSON(ctx, url, q, &event); err != nil {
		return Event{}, err
	}
	return event, nil
}

type urlValues = map[string]string

// getJSON issues one authenticated GET and decodes its body into out.
// Shared by FetchOdds and FetchEventBTTS, which differ only in the path,
// query params, and whether the response is an array or a single object.
func (c *Client) getJSON(ctx context.Context, url string, params urlValues, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("building request: %w", err)
	}
	q := req.URL.Query()
	q.Set("apiKey", c.apiKey)
	for k, v := range params {
		q.Set(k, v)
	}
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("calling the odds api: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("reading response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("the odds api returned %d: %s", resp.StatusCode, body)
	}

	if err := json.Unmarshal(body, out); err != nil {
		return fmt.Errorf("decoding response: %w", err)
	}
	return nil
}
