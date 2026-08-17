// Package config loads the ingestor's runtime configuration from environment
// variables, matching the .env-based workflow used by the ml/ Python scripts.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	SportmonksAPIKey string
	PostgresURL      string

	// PollInterval controls how often the ticker triggers a pull cycle.
	PollInterval time.Duration
	// WindowDays is how far back (from now) to search for finished fixtures
	// on each cycle.
	WindowDays int
	// RecomputeCmd, if set, is executed (via `sh -c`) after a cycle writes
	// at least one new/changed fixture - e.g. "python3 ../ml/build_features.py".
	RecomputeCmd string
}

// Load reads configuration from the environment, first loading a .env file
// if one is found by walking up from the working directory (mirrors
// python-dotenv's upward search used elsewhere in this repo).
func Load() (Config, error) {
	loadDotenvUpward()

	apiKey := os.Getenv("SPORTMONKS_API_KEY")
	if apiKey == "" {
		return Config{}, fmt.Errorf("SPORTMONKS_API_KEY is required")
	}

	pgURL := os.Getenv("POSTGRES_URL")
	if pgURL == "" {
		return Config{}, fmt.Errorf("POSTGRES_URL is required")
	}

	pollInterval, err := durationEnv("INGEST_POLL_INTERVAL", time.Hour)
	if err != nil {
		return Config{}, err
	}

	windowDays, err := intEnv("INGEST_WINDOW_DAYS", 3)
	if err != nil {
		return Config{}, err
	}

	return Config{
		SportmonksAPIKey: apiKey,
		PostgresURL:      pgURL,
		PollInterval:     pollInterval,
		WindowDays:       windowDays,
		RecomputeCmd:     os.Getenv("INGEST_RECOMPUTE_CMD"),
	}, nil
}

func durationEnv(key string, fallback time.Duration) (time.Duration, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return d, nil
}

func intEnv(key string, fallback int) (int, error) {
	raw := os.Getenv(key)
	if raw == "" {
		return fallback, nil
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, fmt.Errorf("invalid %s: %w", key, err)
	}
	return n, nil
}

// loadDotenvUpward walks up from the current directory looking for a .env
// file, so `go run ./cmd/ingestor` works the same whether invoked from
// ingestor-go/ or the repo root. Silently does nothing if none is found -
// in deployed environments (ECS) env vars are set directly.
func loadDotenvUpward() {
	dir, err := os.Getwd()
	if err != nil {
		return
	}
	for i := 0; i < 6; i++ {
		candidate := filepath.Join(dir, ".env")
		if _, err := os.Stat(candidate); err == nil {
			_ = godotenv.Load(candidate)
			return
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return
		}
		dir = parent
	}
}
