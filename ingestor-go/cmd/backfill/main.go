// Command backfill is a one-off tool to pull historical finished fixtures
// into Postgres for a date range further back than the regular ingestor's
// rolling window/gap-sweep would ever reach (e.g. last season), so teams
// early in a new season have enough history for MIN_WINDOW-gated
// predictions (ml/feature_lib.go) right away instead of waiting for the
// current season to accumulate matches.
//
// Usage:
//
//	go run ./cmd/backfill -start 2025-08-01 -end 2026-06-01
//
// Pulls in -chunk-days-sized windows (SportMonks' /fixtures/between endpoint
// isn't paginated by this client - see internal/sportmonks/client.go - so
// chunking keeps each request's result set reasonable) with a short delay
// between chunks to stay polite on the shared API quota (same reasoning as
// ml/fetch_historical_data.py's REQUEST_DELAY_SECONDS).
package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/taikishank/liveedge/ingestor-go/internal/config"
	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
	"github.com/taikishank/liveedge/ingestor-go/internal/store"
)

const dateLayout = "2006-01-02"

func main() {
	startFlag := flag.String("start", "", "start date, inclusive (YYYY-MM-DD)")
	endFlag := flag.String("end", "", "end date, inclusive (YYYY-MM-DD)")
	chunkDays := flag.Int("chunk-days", 15, "days per SportMonks request")
	delay := flag.Duration("delay", 2*time.Second, "delay between chunk requests")
	flag.Parse()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	if *startFlag == "" || *endFlag == "" {
		logger.Error("both -start and -end are required")
		os.Exit(1)
	}
	start, err := time.Parse(dateLayout, *startFlag)
	if err != nil {
		logger.Error("invalid -start date", "error", err)
		os.Exit(1)
	}
	end, err := time.Parse(dateLayout, *endFlag)
	if err != nil {
		logger.Error("invalid -end date", "error", err)
		os.Exit(1)
	}
	if !end.After(start) {
		logger.Error("-end must be after -start")
		os.Exit(1)
	}
	if *chunkDays < 1 {
		logger.Error("-chunk-days must be at least 1")
		os.Exit(1)
	}

	cfg, err := config.Load()
	if err != nil {
		logger.Error("loading config", "error", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	db, err := store.Connect(ctx, cfg.PostgresURL)
	if err != nil {
		logger.Error("connecting to postgres", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	if err := db.Migrate(ctx); err != nil {
		logger.Error("migrating schema", "error", err)
		os.Exit(1)
	}

	client := sportmonks.NewClient(cfg.SportmonksAPIKey)
	// recomputeCmd left empty: this is a one-off bulk load, not worth
	// triggering a feature recompute after every chunk - run it once by
	// hand (or restart the inference service) after the backfill finishes.
	svc := ingest.NewService(client, db, cfg.WindowDays, "", logger)

	logger.Info("starting backfill", "start", start.Format(dateLayout), "end", end.Format(dateLayout), "chunk_days", *chunkDays)

	chunkStart := start
	for chunkStart.Before(end) {
		select {
		case <-ctx.Done():
			logger.Warn("backfill interrupted")
			return
		default:
		}

		chunkEnd := chunkStart.AddDate(0, 0, *chunkDays)
		if chunkEnd.After(end) {
			chunkEnd = end
		}

		svc.RunRange(ctx, chunkStart, chunkEnd)

		chunkStart = chunkEnd
		if chunkStart.Before(end) {
			time.Sleep(*delay)
		}
	}

	logger.Info("backfill complete")
}
