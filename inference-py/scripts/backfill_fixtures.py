"""Loads the 26-season raw archive into Postgres' `fixtures`, so the serving
path reads the history the models were trained on.

scripts/verify_archive_read.py found the gap this closes: `fixtures` held
1,937 completed results spanning one season, against 40,098 in
ml/data/features/results_index.parquet. ingestor-go only ever writes a rolling
window of current fixtures (see UpsertFixtures's comment on its window), so the
table has never held anything older than the day the ingestor first ran. That
is invisible to v2, whose form window is eight matches deep and satisfied by
one season - and fatal to v3, whose `atkdef` ratings are a running summary with
no lookback at all: replaying 5% of the archive produces a rating that is not
the trained quantity and does not error.

The derivation here mirrors ingestor-go/internal/ingest/parse.go exactly,
because the two paths write the same table and a row's provenance must not
change what it says:

  * final score from the CURRENT score entry (type_id 1525), falling back to
    the per-participant goals statistic (type_id 52) - parse.go's finalGoals()
    and build_rating_features.build_index() agree on this pair already;
  * fixtures with no derivable score are skipped, not inserted with NULL
    goals, which is parseFixture()'s `(zero, false)` branch. The ingestor's
    *upcoming* path inserts score-less rows, but those are fixtures that have
    not kicked off yet; a 2007 fixture with no score is an archive hole, and
    writing it would put a phantom fixture in front of every query whose only
    guard is an IS NOT NULL on the goals;
  * `starting_at` is parsed as naive UTC, matching parse.go's
    time.Parse("2006-01-02 15:04:05", ...) with no zone - the provider's
    convention, and the parquet's.

The upsert clause is the ingestor's own, including its
`WHERE fixtures.raw IS DISTINCT FROM EXCLUDED.raw` guard, so re-running this
is a no-op on rows already correct and this script can never fight the live
ingestor over the current season: whichever wrote last wins, and both derive
the same values from the same payload.

Usage (Postgres running on the host; .env at the repo root supplies
POSTGRES_URL). Dry-run first - it stages the whole load, reports what would
change, and rolls back:

    ./inference-py/bin/python inference-py/scripts/backfill_fixtures.py
    ./inference-py/bin/python inference-py/scripts/backfill_fixtures.py --apply

Then re-run scripts/verify_archive_read.py, which should go to PASS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg

from app.config import POSTGRES_URL

RAW_DIR = Path(__file__).resolve().parents[2] / "ml" / "data" / "raw" / "all_fixtures"

# Mirrors ingest/parse.go's finalScoreTypeID / goalsStatTypeID, and
# build_rating_features.py's constants of the same name.
FINAL_SCORE_TYPE_ID = 1525
GOALS_STAT_TYPE_ID = 52

#: How often to report progress while streaming ~440MB of JSON into the
#: staging table.
PROGRESS_EVERY = 10000


def parse_fixture(payload: dict) -> tuple | None:
    """parse.go's parseFixture(), in Python. Returns None where it returns
    (zero, false): no home/away participant, or no derivable final score."""
    try:
        home = next(p for p in payload["participants"] if p["meta"]["location"] == "home")
        away = next(p for p in payload["participants"] if p["meta"]["location"] == "away")
    except (StopIteration, KeyError):
        return None

    final = {
        s["score"]["participant"]: s["score"]["goals"]
        for s in payload.get("scores", [])
        if s.get("type_id") == FINAL_SCORE_TYPE_ID
    }
    if "home" in final and "away" in final:
        home_goals, away_goals = final["home"], final["away"]
    else:
        goals = {
            s["participant_id"]: s["data"]["value"]
            for s in payload.get("statistics", [])
            if s.get("type_id") == GOALS_STAT_TYPE_ID
            and s.get("data", {}).get("value") is not None
        }
        if home["id"] not in goals or away["id"] not in goals:
            return None
        home_goals, away_goals = goals[home["id"]], goals[away["id"]]

    starting_at = datetime.strptime(
        payload["starting_at"], "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=timezone.utc)
    home_goals, away_goals = int(home_goals), int(away_goals)
    result = ("home_win" if home_goals > away_goals
              else "away_win" if away_goals > home_goals else "draw")

    return (
        payload["id"], payload["league_id"], starting_at,
        home["id"], away["id"], home["name"], away["name"],
        home_goals, away_goals, result, json.dumps(payload),
    )


def iter_rows():
    """Parsed rows, one per archive file, skipping the unscorable."""
    skipped = 0
    seen: set[int] = set()
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        row = parse_fixture(payload)
        if row is None:
            skipped += 1
            continue
        # The staging table's primary key would reject a repeat, and a repeat
        # is a fact about the archive worth reporting rather than a crash.
        if row[0] in seen:
            skipped += 1
            continue
        seen.add(row[0])
        yield row
    print(f"  skipped {skipped:,} files with no derivable final score (or a duplicate id)")


#: The ingestor's own upsert (store/postgres.go UpsertFixtures), sourced from a
#: staging table instead of one VALUES tuple per fixture.
UPSERT = """
    INSERT INTO fixtures (
        fixture_id, league_id, starting_at, home_id, away_id,
        home_name, away_name, home_goals, away_goals, result, raw
    )
    SELECT fixture_id, league_id, starting_at, home_id, away_id,
           home_name, away_name, home_goals, away_goals, result, raw
    FROM staging_fixtures
    ON CONFLICT (fixture_id) DO UPDATE SET
        home_goals  = EXCLUDED.home_goals,
        away_goals  = EXCLUDED.away_goals,
        result      = EXCLUDED.result,
        raw         = EXCLUDED.raw,
        ingested_at = now()
    WHERE fixtures.raw IS DISTINCT FROM EXCLUDED.raw
"""

STAGING_DDL = """
    CREATE TEMP TABLE staging_fixtures (
        fixture_id  bigint PRIMARY KEY,
        league_id   bigint NOT NULL,
        starting_at timestamptz NOT NULL,
        home_id     bigint NOT NULL,
        away_id     bigint NOT NULL,
        home_name   text NOT NULL,
        away_name   text NOT NULL,
        home_goals  integer,
        away_goals  integer,
        result      text,
        raw         jsonb NOT NULL
    ) ON COMMIT DROP
"""

COPY_SQL = """
    COPY staging_fixtures (
        fixture_id, league_id, starting_at, home_id, away_id,
        home_name, away_name, home_goals, away_goals, result, raw
    ) FROM STDIN
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="commit the upsert (default: stage, report, roll back)")
    args = parser.parse_args()

    if not RAW_DIR.is_dir():
        print(f"no raw archive at {RAW_DIR}", file=sys.stderr)
        return 1

    print(f"archive:  {RAW_DIR}")
    print(f"postgres: {POSTGRES_URL.split('@')[-1]}")
    print(f"mode:     {'APPLY (writes)' if args.apply else 'dry run (rolls back)'}\n")

    started = time.monotonic()
    staged = 0
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM fixtures")
        before_all = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM fixtures "
                    "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL")
        before_done = cur.fetchone()[0]
        print(f"  before: {before_all:,} rows, {before_done:,} completed")

        cur.execute(STAGING_DDL)
        # Staged rather than upserted row by row: one COPY plus one
        # INSERT ... SELECT is a single statement the server plans once, and it
        # leaves the dry run an honest rehearsal of the real write rather than a
        # different code path.
        with cur.copy(COPY_SQL) as copy:
            for row in iter_rows():
                copy.write_row(row)
                staged += 1
                if staged % PROGRESS_EVERY == 0:
                    print(f"  staged {staged:,}...", flush=True)
        print(f"  staged {staged:,} fixtures in {time.monotonic() - started:.1f}s")

        cur.execute(UPSERT)
        changed = cur.rowcount
        cur.execute("SELECT count(*) FROM fixtures")
        after_all = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM fixtures "
                    "WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL")
        after_done = cur.fetchone()[0]

        print(f"\n  rows inserted or updated: {changed:,}")
        print(f"  after:  {after_all:,} rows ({after_all - before_all:+,}),"
              f" {after_done:,} completed ({after_done - before_done:+,})")

        if args.apply:
            conn.commit()
            print("\n  committed.")
        else:
            conn.rollback()
            print("\n  rolled back - nothing was written. Re-run with --apply.")

    print(f"  total {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
