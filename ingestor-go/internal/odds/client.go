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
// the given sport, from eu bookmakers. The free tier is metered per call, so
// callers should poll infrequently (see config.OddsPollInterval).
func (c *Client) FetchOdds(ctx context.Context, sportKey string) ([]Event, error) {
	url := fmt.Sprintf("%s/sports/%s/odds", c.baseURL, sportKey)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("building request: %w", err)
	}
	q := req.URL.Query()
	q.Set("apiKey", c.apiKey)
	q.Set("regions", "eu")
	q.Set("markets", "h2h")
	q.Set("oddsFormat", "decimal")
	req.URL.RawQuery = q.Encode()

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("calling the odds api: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response body: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("the odds api returned %d: %s", resp.StatusCode, body)
	}

	var events []Event
	if err := json.Unmarshal(body, &events); err != nil {
		return nil, fmt.Errorf("decoding response: %w", err)
	}
	return events, nil
}
