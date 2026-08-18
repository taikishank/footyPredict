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
	"github.com/taikishank/liveedge/ingestor-go/internal/sportmonks"
	"github.com/taikishank/liveedge/ingestor-go/internal/store"
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

	logger.Info("ingestor starting",
		"poll_interval", cfg.PollInterval.String(),
		"window_days", cfg.WindowDays,
		"live_poll_interval", cfg.LivePollInterval.String(),
	)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		svc.Run(ctx, cfg.PollInterval)
	}()
	go func() {
		defer wg.Done()
		liveSvc.Run(ctx, cfg.LivePollInterval)
	}()
	wg.Wait()

	logger.Info("ingestor stopped")
}
