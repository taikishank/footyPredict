"""Refreshes fixtures in ml/data/raw/all_fixtures/ from Postgres `fixtures.raw`.

The archive is what ml/build_features.py reads, and it's the only immutable
copy of the training corpus - Postgres mutates (fixtures get re-ingested, and
a re-ingest has been observed to change season_id on already-played matches).
Fixtures fetched before kickoff land in the archive as stubs: no final score
and no statistics, so parse_fixture() skips them and they never reach the
feature table. This pulls the finished payload back over those stubs.

Only writes a fixture whose DB payload is actually complete (two participants,
both final-score entries, non-empty statistics), so a stub is never replaced
by another stub, and a good archive file is never downgraded.

Run:
    ./inference-py/bin/python ml/refresh_raw_archive.py 2026-09-12
    ./inference-py/bin/python ml/refresh_raw_archive.py 2026-09-12 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ARCHIVE_DIR = Path(__file__).parent / "data" / "raw" / "all_fixtures"
FINAL_SCORE_TYPE_ID = 1525


def is_complete(raw: dict) -> tuple[bool, str]:
    """Whether a payload carries everything build_features.py needs, both as a
    labelled training row and as history for later fixtures."""
    participants = raw.get("participants") or []
    if len(participants) != 2:
        return False, f"{len(participants)} participants"
    locations = {p.get("meta", {}).get("location") for p in participants}
    if locations != {"home", "away"}:
        return False, f"participant locations {sorted(locations)}"

    finals = {
        s["score"]["participant"]
        for s in (raw.get("scores") or [])
        if s.get("type_id") == FINAL_SCORE_TYPE_ID
    }
    if finals != {"home", "away"}:
        return False, f"final-score entries {sorted(finals)}"

    if not (raw.get("statistics") or []):
        return False, "no statistics"
    return True, ""


def archive_state(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return "complete" if is_complete(json.loads(path.read_text()))[0] else "stub"
    except (json.JSONDecodeError, OSError) as exc:
        return f"unreadable ({exc})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day", help="UTC-ish calendar day to refresh, e.g. 2026-09-12")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    with psycopg.connect(os.environ["POSTGRES_URL"]) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id, home_name, away_name, result, raw
            FROM fixtures
            WHERE starting_at >= %s::date AND starting_at < %s::date + 1
            ORDER BY starting_at, fixture_id
            """,
            (args.day, args.day),
        )
        rows = cur.fetchall()

    written = skipped_incomplete = already_ok = 0
    for fixture_id, home, away, result, raw in rows:
        path = ARCHIVE_DIR / f"{fixture_id}.json"
        before = archive_state(path)
        ok, why = is_complete(raw)
        label = f"{fixture_id}  {home} vs {away}"

        if not ok:
            print(f"  SKIP     {label}: db payload incomplete ({why}); archive is {before}")
            skipped_incomplete += 1
            continue
        if before == "complete":
            print(f"  ok       {label}: archive already complete")
            already_ok += 1
            continue
        if args.dry_run:
            print(f"  WOULD    {label}: {before} -> complete (result={result})")
            written += 1
            continue

        # Write via a temp file in the same directory, then rename, so an
        # interrupted run can't leave a half-written payload behind.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
        tmp.replace(path)
        print(f"  WROTE    {label}: {before} -> complete (result={result})")
        written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\n{len(rows)} fixture(s) on {args.day}: {verb} {written}, "
        f"{already_ok} already complete, {skipped_incomplete} skipped as incomplete",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
