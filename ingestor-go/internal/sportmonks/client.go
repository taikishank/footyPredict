// Package sportmonks is a minimal client for the SportMonks v3 Football API,
// scoped to what the batch ingestor needs: pulling finished fixtures in a
// date window. See ml/fetch_historical_data.py for the equivalent Python
// client used during Phase 1's historical backfill.
package sportmonks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"
)

const (
	defaultBaseURL = "https://api.sportmonks.com/v3/football"
	dateLayout     = "2006-01-02"

	// rateLimitFloor is the minimum remaining-requests-this-hour count
	// (as reported by SportMonks' own rate_limit field) below which the
	// client refuses to make further requests, leaving headroom before
	// the account's hourly quota is actually exhausted.
	rateLimitFloor = 500
)

// ErrRateLimited is returned instead of making a request when the last known
// SportMonks rate_limit.remaining count is below rateLimitFloor.
var ErrRateLimited = errors.New("sportmonks: remaining hourly requests below floor, skipping call")

type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client

	mu            sync.Mutex
	haveRateLimit bool
	remaining     int
}

func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:     apiKey,
		baseURL:    defaultBaseURL,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

// FetchFinishedBetween returns all finished fixtures with a starting_at date
// between start and end (inclusive), across all leagues the API key covers.
// It refuses to call the API at all if the last known rate limit is too low;
// see ErrRateLimited.
func (c *Client) FetchFinishedBetween(ctx context.Context, start, end time.Time) ([]Fixture, error) {
	if c.rateLimitExhausted() {
		return nil, ErrRateLimited
	}

	url := fmt.Sprintf("%s/fixtures/between/%s/%s", c.baseURL, start.Format(dateLayout), end.Format(dateLayout))

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	q := req.URL.Query()
	q.Set("api_token", c.apiKey)
	q.Set("include", "participants;scores;statistics;state")
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
		if f.IsFinished() {
			fixtures = append(fixtures, f)
		}
	}
	return fixtures, nil
}

func (c *Client) rateLimitExhausted() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.haveRateLimit && c.remaining < rateLimitFloor
}

func (c *Client) updateRateLimit(rl *RateLimit) {
	if rl == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.haveRateLimit = true
	c.remaining = rl.Remaining
}
