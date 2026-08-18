package sportmonks

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// FetchLive returns all fixtures SportMonks currently considers in-play,
// via the livescores/inplay endpoint. It shares the same account-wide rate
// limit floor as FetchFinishedBetween - see rateLimitFloor.
func (c *Client) FetchLive(ctx context.Context) ([]Fixture, error) {
	if c.rateLimitExhausted() {
		return nil, ErrRateLimited
	}

	url := fmt.Sprintf("%s/livescores/inplay", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	q := req.URL.Query()
	q.Set("api_token", c.apiKey)
	q.Set("include", "participants;scores;statistics;state;events")
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling sportmonks: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("sportmonks returned %d: %s", resp.StatusCode, body)
	}

	var parsed fixturesResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	c.updateRateLimit(parsed.RateLimit)

	fixtures := make([]Fixture, 0, len(parsed.Data))
	for _, raw := range parsed.Data {
		var f Fixture
		if err := json.Unmarshal(raw, &f); err != nil {
			return nil, fmt.Errorf("decoding fixture: %w", err)
		}
		f.Raw = raw
		fixtures = append(fixtures, f)
	}
	return fixtures, nil
}
