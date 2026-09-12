#!/usr/bin/env bash
# Starts the full local dev stack: frontend (ng serve), inference-py
# (uvicorn), and both ingestor-go binaries (ingestor, live-consumer).
# Assumes Postgres is already running locally and .env at the repo root
# has POSTGRES_URL / SPORTMONKS_API_KEY / ODDS_API_KEY / AWS_* set.
#
# Usage: ./run-dev.sh
# Stop everything with Ctrl+C.

set -euo pipefail

# Job control, so every background service lands in its own process group
# (pgid == $!). Cleanup then signals the whole group: `go run` spawns the
# compiled binary as a child, and killing only `go run` orphans it.
set -m

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/.devlogs"
mkdir -p "$LOG_DIR"

set -a
source "$ROOT_DIR/.env"
set +a

# Ports the stack needs free before it starts. A leftover process from an
# earlier run holding one of these makes the new service die on startup.
declare -A PORTS=( [4200]=frontend [8000]=inference-py )

check_ports() {
    local blocked=() port
    for port in "${!PORTS[@]}"; do
        if ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
            blocked+=("$port (${PORTS[$port]})")
            echo "port $port is already in use by:"
            ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2
        fi
    done
    if ((${#blocked[@]})); then
        echo
        echo "refusing to start; free the port(s) first:"
        for port in "${!PORTS[@]}"; do
            ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN && echo "  fuser -k $port/tcp"
        done
        exit 1
    fi
}

pgids=()
names=()

cleanup() {
    trap - EXIT INT TERM
    ((${#pgids[@]})) || return 0
    echo
    echo "stopping dev stack..."
    local pgid
    for pgid in "${pgids[@]}"; do
        kill -TERM -- "-$pgid" 2>/dev/null || true
    done
    # Give them a moment to shut down cleanly, then insist.
    local waited=0
    while ((waited < 50)); do
        local alive=0
        for pgid in "${pgids[@]}"; do
            kill -0 -- "-$pgid" 2>/dev/null && alive=1
        done
        ((alive)) || break
        sleep 0.1
        ((waited++))
    done
    for pgid in "${pgids[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
    done
}
trap cleanup EXIT INT TERM

start() {
    local name="$1" dir="$2"
    shift 2
    echo "starting $name (log: $LOG_DIR/$name.log)"
    (cd "$dir" && exec "$@") >"$LOG_DIR/$name.log" 2>&1 </dev/null &
    pgids+=("$!")
    names+=("$name")
}

# Fail loudly if a service died on startup, instead of tailing logs forever
# next to a stack that is quietly missing a backend.
verify_started() {
    sleep 3
    local i dead=()
    for i in "${!pgids[@]}"; do
        kill -0 "${pgids[$i]}" 2>/dev/null || dead+=("${names[$i]}")
    done
    if ((${#dead[@]})); then
        echo
        local name
        for name in "${dead[@]}"; do
            echo "!! $name exited during startup; last lines of $LOG_DIR/$name.log:"
            tail -n 15 "$LOG_DIR/$name.log" | sed 's/^/     /'
            echo
        done
        echo "stack is incomplete - shutting the rest down."
        exit 1
    fi
}

check_ports

start frontend "$ROOT_DIR/frontend" npm run start
start inference-py "$ROOT_DIR/inference-py" bin/uvicorn app.main:app --reload
start ingestor "$ROOT_DIR/ingestor-go" go run ./cmd/ingestor
start live-consumer "$ROOT_DIR/ingestor-go" go run ./cmd/live-consumer

verify_started

echo
echo "all services started. tailing logs (Ctrl+C to stop everything)."
echo

tail -n +1 -f "$LOG_DIR"/*.log
