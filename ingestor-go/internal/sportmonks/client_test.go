package sportmonks

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestFetchFinishedBetween_FiltersUnfinishedAndSendsExpectedRequest(t *testing.T) {
	var gotPath, gotAPIToken, gotInclude string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAPIToken = r.URL.Query().Get("api_token")
		gotInclude = r.URL.Query().Get("include")

		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"data": [
				{
					"id": 1, "league_id": 8, "starting_at": "2026-01-10 15:00:00",
					"state": {"short_name": "FT"},
					"participants": [], "scores": [], "statistics": []
				},
				{
					"id": 2, "league_id": 8, "starting_at": "2026-01-11 15:00:00",
					"state": {"short_name": "NS"},
					"participants": [], "scores": [], "statistics": []
				},
				{
					"id": 3, "league_id": 8, "starting_at": "2026-01-11 15:00:00",
					"state": {"short_name": "POSTP"},
					"participants": [], "scores": [], "statistics": []
				}
			]
		}`))
	}))
	defer server.Close()

	client := NewClient("test-key")
	client.baseURL = server.URL
	client.httpClient = server.Client()

	start := time.Date(2026, 1, 8, 0, 0, 0, 0, time.UTC)
	end := time.Date(2026, 1, 11, 0, 0, 0, 0, time.UTC)

	fixtures, err := client.FetchFinishedBetween(context.Background(), start, end)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(fixtures) != 1 {
		t.Fatalf("got %d fixtures, want 1 (only FT should pass the finished filter)", len(fixtures))
	}
	if fixtures[0].ID != 1 {
		t.Fatalf("got fixture id %d, want 1", fixtures[0].ID)
	}

	if !strings.HasSuffix(gotPath, "/fixtures/between/2026-01-08/2026-01-11") {
		t.Errorf("got path %q, want suffix /fixtures/between/2026-01-08/2026-01-11", gotPath)
	}
	if gotAPIToken != "test-key" {
		t.Errorf("got api_token %q, want test-key", gotAPIToken)
	}
	if gotInclude != "participants;scores;statistics;state" {
		t.Errorf("got include %q", gotInclude)
	}
}

func TestFetchFinishedBetween_SkipsCallWhenRateLimitBelowFloor(t *testing.T) {
	calls := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data": [], "rate_limit": {"remaining": 499, "resets_in_seconds": 1200}}`))
	}))
	defer server.Close()

	client := NewClient("test-key")
	client.baseURL = server.URL
	client.httpClient = server.Client()

	// First call succeeds and records the low remaining count.
	if _, err := client.FetchFinishedBetween(context.Background(), time.Now(), time.Now()); err != nil {
		t.Fatalf("unexpected error on first call: %v", err)
	}
	if calls != 1 {
		t.Fatalf("got %d calls after first fetch, want 1", calls)
	}

	// Second call should be skipped locally without hitting the server.
	_, err := client.FetchFinishedBetween(context.Background(), time.Now(), time.Now())
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("got err %v, want ErrRateLimited", err)
	}
	if calls != 1 {
		t.Fatalf("got %d calls after second fetch, want still 1 (should not have called server)", calls)
	}
}

func TestFetchFinishedBetween_ProceedsWhenRateLimitAtOrAboveFloor(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data": [], "rate_limit": {"remaining": 500, "resets_in_seconds": 1200}}`))
	}))
	defer server.Close()

	client := NewClient("test-key")
	client.baseURL = server.URL
	client.httpClient = server.Client()

	if _, err := client.FetchFinishedBetween(context.Background(), time.Now(), time.Now()); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, err := client.FetchFinishedBetween(context.Background(), time.Now(), time.Now()); err != nil {
		t.Fatalf("unexpected error on second call: %v", err)
	}
}

func TestFetchFinishedBetween_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		w.Write([]byte(`{"message": "invalid api key"}`))
	}))
	defer server.Close()

	client := NewClient("bad-key")
	client.baseURL = server.URL

	_, err := client.FetchFinishedBetween(context.Background(), time.Now(), time.Now())
	if err == nil {
		t.Fatal("expected error for non-200 response")
	}
}
