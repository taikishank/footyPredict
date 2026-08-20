// Command ingestor periodically pulls recently completed fixtures from
// SportMonks and writes them to Postgres (LiveEdge Phase 2 batch pipeline).
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"sync"
	"syscall"

	"github.com/taikishank/liveedge/ingestor-go/internal/config"
	"github.com/taikishank/liveedge/ingestor-go/internal/ingest"
	"github.com/taikishank/liveedge/ingestor-go/internal/live"
	"github.com/taikishank/liveedge/ingestor-go/internal/odds"
	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
	"github.com/taikishank/liveedge/ingestor-go/internal/store"
	"github.com/taikishank/liveedge/ingestor-go/internal/upcoming"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

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
	svc := ingest.NewService(client, db, cfg.WindowDays, cfg.RecomputeCmd, logger)

	liveSvc := live.NewService(client, db, live.NewLocalPublisher(logger), logger)
	upcomingSvc := upcoming.NewService(client, db, cfg.UpcomingWindowDays, logger)

	logger.Info("ingestor starting",
		"poll_interval", cfg.PollInterval.String(),
		"window_days", cfg.WindowDays,
		"live_poll_interval", cfg.LivePollInterval.String(),
		"upcoming_poll_interval", cfg.UpcomingPollInterval.String(),
		"upcoming_window_days", cfg.UpcomingWindowDays,
		"odds_poll_interval", cfg.OddsPollInterval.String(),
	)

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		svc.Run(ctx, cfg.PollInterval)
	}()
	go func() {
		defer wg.Done()
		liveSvc.Run(ctx, cfg.LivePollInterval)
	}()
	go func() {
		defer wg.Done()
		upcomingSvc.Run(ctx, cfg.UpcomingPollInterval)
	}()

	if cfg.OddsAPIKey == "" {
		logger.Warn("ODDS_API_KEY not set, skipping odds poller")
	} else {
		oddsClient := odds.NewClient(cfg.OddsAPIKey)
		oddsSvc := odds.NewService(oddsClient, db, db, cfg.UpcomingWindowDays, logger)
		wg.Add(1)
		go func() {
			defer wg.Done()
			oddsSvc.Run(ctx, cfg.OddsPollInterval)
		}()
	}
	wg.Wait()

	logger.Info("ingestor stopped")
}
