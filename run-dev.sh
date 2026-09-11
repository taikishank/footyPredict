#!/usr/bin/env bash
# Starts the full local dev stack: frontend (ng serve), inference-py
# (uvicorn), and both ingestor-go binaries (ingestor, live-consumer).
# Assumes Postgres is already running locally and .env at the repo root
# has POSTGRES_URL / SPORTMONKS_API_KEY / ODDS_API_KEY / AWS_* set.
#
# Usage: ./run-dev.sh
# Stop everything with Ctrl+C.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/.devlogs"
mkdir -p "$LOG_DIR"

set -a
source "$ROOT_DIR/.env"
set +a

pids=()

cleanup() {
    echo
    echo "stopping dev stack..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

start() {
    local name="$1" dir="$2"
    shift 2
    echo "starting $name (log: $LOG_DIR/$name.log)"
    (cd "$dir" && exec "$@") >"$LOG_DIR/$name.log" 2>&1 &
    pids+=("$!")
}

start frontend "$ROOT_DIR/frontend" npm run start
start inference-py "$ROOT_DIR/inference-py" bin/uvicorn app.main:app --reload
start ingestor "$ROOT_DIR/ingestor-go" go run ./cmd/ingestor
start live-consumer "$ROOT_DIR/ingestor-go" go run ./cmd/live-consumer

echo
echo "all services started. tailing logs (Ctrl+C to stop everything)."
echo

tail -n +1 -f "$LOG_DIR"/*.log
