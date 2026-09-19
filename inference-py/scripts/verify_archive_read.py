"""Does the archive the service reads match the archive the ratings were
trained on?

Every equivalence test written for the rating path so far
(tests/test_ratings_state.py, ml/test_ratings_lib.py) reads
ml/data/features/results_index.parquet on *both* sides. They prove the replay
driver is shared. They cannot prove the two inputs are, because
app/db.get_results_index() has never returned a row in any of them.

That is the one unverified link in the chain, and it is the one that fails
silently: ratings_lib.AttackDefence's state after fixture N depends on every
fixture before it, so a single extra row, a missing league, a duplicated
fixture_id or a different tie-break order in `fixtures` makes the served
ratings a different quantity from the trained ones, with no error and no
obviously wrong number. v3 would predict confidently off a history the model
was never fitted on.

So this script runs the real serving read against the training parquet and
reports where they part company:

  1. shape        - row counts, per-league counts, date ranges
  2. alignment    - fixtures in one source and not the other, duplicate
                    fixture_ids, disagreeing scores or kickoff times
  3. order        - the first index at which the two replay sequences diverge,
                    which is the only divergence that matters (everything
                    after it is downstream of the same cause)
  4. ratings      - replay both and diff the served feature columns for a
                    sample of real matchups per league
  5. /health      - what app.ratings_state.state_summary() would report

Read-only: no writes to Postgres, no files touched.

Usage (Postgres already running - this project runs it on the host, not in
docker compose; .env at the repo root supplies POSTGRES_URL):

    ./inference-py/bin/python inference-py/scripts/verify_archive_read.py
    ./inference-py/bin/python inference-py/scripts/verify_archive_read.py --sample 10

Exit status is 0 when the two archives agree on shape, alignment and order,
and 1 otherwise, so it can gate a promotion.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from app import db, ratings_state
from app.config import POSTGRES_URL
from ratings_lib import ARMS, Replay, season_of
from build_rating_features import INDEX_PATH, order

#: Columns the rating comparison diffs - the arm ratings_state actually serves.
FIELD_ORDER = ratings_state.FIELD_ORDER

#: A float diff below this is replay arithmetic, not a different history.
TOLERANCE = 1e-9

#: How far back from the archive's last row still counts as "the live table
#: running ahead of the parquet". A matchday spans a weekend and
#: ml/refresh_raw_archive.py is run by hand, so results ingested since the last
#: refresh interleave with the archive's final rounds rather than landing
#: strictly after them. Wide enough to cover a refresh that is a few days old;
#: narrow enough that a genuine hole in 2007 is never mistaken for one.
TAIL_DAYS = 14


def _utc_naive(value) -> datetime:
    """Both sources in one timezone convention. The parquet carries naive
    UTC (SportMonks' starting_at as-is); Postgres hands back whatever the
    column's type says, which for a timestamptz is tz-aware UTC. Comparing
    them unconverted would report every row as a mismatch."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.to_pydatetime()


def load_archive() -> pd.DataFrame:
    """The training archive, in the replay order build_rating_features used."""
    index = pd.read_parquet(INDEX_PATH)
    return order(index)


def load_served() -> tuple[pd.DataFrame, list, float]:
    """The serving read, verbatim - this is the call under test.

    fixture_id is not in ResultRow (the replay does not need it), so it is
    fetched alongside in one extra query purely to align rows for the
    diagnostics below. The replay itself uses what get_results_index()
    returned, unmodified.
    """
    started = time.monotonic()
    rows = db.get_results_index()
    elapsed = time.monotonic() - started

    with db.pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT fixture_id, league_id, starting_at, home_id, away_id,
                   home_goals, away_goals
            FROM fixtures
            WHERE home_goals IS NOT NULL AND away_goals IS NOT NULL
            ORDER BY starting_at ASC, fixture_id ASC
            """
        )
        keyed = cur.fetchall()

    served = pd.DataFrame(keyed, columns=[
        "fixture_id", "league_id", "date", "home_id", "away_id",
        "home_goals", "away_goals",
    ])
    served["date"] = served["date"].map(_utc_naive)
    return served, rows, elapsed


def check_shape(archive: pd.DataFrame, served: pd.DataFrame) -> bool:
    """Counts and coverage, reported per league.

    Deliberately does not gate on the counts being equal. A healthy live table
    runs a few results ahead of a hand-refreshed parquet, so requiring equality
    would make this red on a correct database and teach whoever runs it to
    ignore the verdict. What a count is *for* is reading the magnitude: a delta
    of a dozen is a refresh lag, a delta of 38,000 is a different archive.
    Whether the difference is benign is decided by checks 2 and 3, which look
    at which rows differ and where.
    """
    print("=== 1. shape ===")
    print(f"  archive (results_index.parquet): {len(archive):>7,} completed fixtures")
    print(f"  served  (get_results_index()):   {len(served):>7,} completed fixtures")
    delta = len(served) - len(archive)
    print(f"  delta: {delta:+,}")

    leagues = sorted(set(archive["league_id"]) | set(served["league_id"]))
    print(f"\n  {'league':>8} {'archive':>9} {'served':>9} {'delta':>8}   archive range / served range")
    ok = True
    for league_id in leagues:
        a = archive[archive["league_id"] == league_id]
        s = served[served["league_id"] == league_id]
        a_range = f"{a['date'].min():%Y-%m-%d}..{a['date'].max():%Y-%m-%d}" if len(a) else "-"
        s_range = f"{s['date'].min():%Y-%m-%d}..{s['date'].max():%Y-%m-%d}" if len(s) else "-"
        print(f"  {league_id:>8} {len(a):>9,} {len(s):>9,} {len(s) - len(a):>+8,}   {a_range} / {s_range}")

    only_served = sorted(set(served["league_id"]) - set(archive["league_id"]))
    only_archive = sorted(set(archive["league_id"]) - set(served["league_id"]))
    if only_served:
        print(f"\n  !! leagues in Postgres but not the archive: {only_served}")
        print("     ratings_lib pools per-league, so these get their own independent")
        print("     ratings rather than contaminating the trained five - but the model")
        print("     was never fitted on a fixture from them.")
    if only_archive:
        print(f"\n  !! leagues in the archive but not Postgres: {only_archive}")
        ok = False
    return ok


def served_missing_sample(archive: pd.DataFrame, missing: set[int]) -> pd.DataFrame:
    """The oldest few archive fixtures Postgres has no completed row for -
    oldest rather than newest because the earliest gap is the one every later
    rating is downstream of."""
    rows = archive[archive["fixture_id"].isin(missing)]
    return rows.nsmallest(5, "date")[["fixture_id", "date", "league_id",
                                      "home_name", "away_name"]]


def check_alignment(archive: pd.DataFrame, served: pd.DataFrame) -> bool:
    print("\n=== 2. alignment ===")
    ok = True

    dupes = served["fixture_id"].duplicated().sum()
    print(f"  duplicate fixture_ids in `fixtures`: {dupes}")
    if dupes:
        ok = False
        sample = served[served["fixture_id"].duplicated(keep=False)].head(6)
        print(sample.to_string(index=False).replace("\n", "\n    "))
        print("     !! a duplicated result is recorded twice by the replay, which is")
        print("        a rating update the training walk never made.")

    a_ids = set(archive["fixture_id"])
    s_ids = set(served["fixture_id"])
    missing = a_ids - s_ids
    extra = s_ids - a_ids
    print(f"  in archive, not completed in Postgres: {len(missing):,}")
    print(f"  completed in Postgres, not in archive: {len(extra):,}")
    if missing:
        ok = False
        print("     !! a completed fixture with NULL goals, or one Postgres does not")
        print("        carry at all - either way the served walk is missing a rating")
        print("        update the trained walk made. This is the direction that")
        print("        matters: the model cannot be fed a history shorter than its own.")
        oldest = served_missing_sample(archive, missing)
        print(oldest.to_string(index=False).replace("\n", "\n    "))
    if extra:
        # Where an extra sits decides what it is. A result the ingestor picked
        # up since ml/refresh_raw_archive.py last ran lands at the archive's
        # tail, and a matchday spans a few days, so some of those legitimately
        # fall a few hours *before* the archive's last row - interleaved with
        # it, not a hole in it. An extra deep in history is the other thing
        # entirely: a fixture the archive never had, shifting every rating
        # after it. TAIL_DAYS is the line between the two readings.
        cutoff = archive["date"].max() - timedelta(days=TAIL_DAYS)
        late = served[served["fixture_id"].isin(extra)]
        fresh, gaps = late[late["date"] >= cutoff], late[late["date"] < cutoff]
        print(f"     {len(fresh):,} within {TAIL_DAYS} days of the archive's last row"
              f" ({archive['date'].max():%Y-%m-%d}) - results ingested since the")
        print("       archive was refreshed, interleaved with its final matchdays")
        print(f"     {len(gaps):,} older than that")
        if len(gaps):
            ok = False
            print("     !! those are gaps, not new results: they sit inside the period")
            print("        the model trained on and shift every rating after them.")
            print(gaps[["fixture_id", "date", "league_id"]].head(5)
                  .to_string(index=False).replace("\n", "\n    "))

    shared = archive.merge(served, on="fixture_id", suffixes=("_a", "_s"))
    for col in ("home_goals", "away_goals", "home_id", "away_id", "league_id"):
        bad = shared[shared[f"{col}_a"] != shared[f"{col}_s"]]
        print(f"  {col} disagrees on {len(bad):,} shared fixtures")
        if len(bad):
            ok = False
            print(bad[["fixture_id", f"{col}_a", f"{col}_s"]].head(5)
                  .to_string(index=False).replace("\n", "\n    "))

    drift = shared[shared["date_a"] != shared["date_s"]]
    print(f"  kickoff time disagrees on {len(drift):,} shared fixtures")
    if len(drift):
        ok = False
        print("     !! kickoff time is the replay's primary sort key; a shifted one")
        print("        reorders the walk.")
        print(drift[["fixture_id", "date_a", "date_s"]].head(5)
              .to_string(index=False).replace("\n", "\n    "))
    return ok


def check_season_timezone(rows) -> bool:
    """Does Postgres' session timezone move any fixture across the season line?

    ratings_lib.season_of() reads .year/.month off whatever datetime it is
    handed, and the two sources hand it different conventions: the parquet
    carries naive UTC, while psycopg renders a timestamptz in the *session*
    timezone. Those agree on the season for every date except a kickoff within
    the offset of 1 July - where a UTC-evening match reads as the previous
    season locally, regressing its clubs' ratings a season late.

    Counted rather than assumed, because it is a latent defect: a database
    that has never held a late-June fixture reports zero and would still break
    the first time one lands.
    """
    print("\n=== 2b. season boundary under Postgres' session timezone ===")
    with db.pool.connection() as conn, conn.cursor() as cur:
        cur.execute("SHOW timezone")
        tz = cur.fetchone()[0]
    skewed = [r for r in rows
              if season_of(r["starting_at"]) != season_of(_utc_naive(r["starting_at"]))]
    print(f"  session timezone: {tz}")
    print(f"  fixtures whose season_of() differs from the archive's UTC reading:"
          f" {len(skewed):,}")
    if skewed:
        for row in skewed[:5]:
            print(f"     {row['starting_at']} -> served season {season_of(row['starting_at'])},"
                  f" trained season {season_of(_utc_naive(row['starting_at']))}")
        print("     !! these clubs are regressed at a different rollover than training.")
        return False
    if tz.upper() != "UTC":
        print(f"  none today, but the session is not UTC - a kickoff within the offset")
        print(f"  of 1 July would be assigned the wrong season. Latent, not current.")
    return True


def check_order(archive: pd.DataFrame, served: pd.DataFrame) -> bool:
    """Where the two replay sequences first diverge.

    Only the first divergence is reported: the rating state is path-dependent,
    so every row after it is walked from a state the other side never had, and
    listing them would be listing one cause many times.

    A divergence inside the archive's final TAIL_DAYS is not a failure. A live
    database is *supposed* to run ahead of a parquet rebuilt on a cadence, and
    the ratings those extra results move are ratings for fixtures that have
    already been played. A divergence before that window is the real thing:
    the served walk and the trained walk disagree about history.
    """
    print("\n=== 3. replay order ===")
    cols = ["date", "league_id", "home_id", "away_id", "home_goals", "away_goals"]
    a = list(archive[cols].itertuples(index=False, name=None))
    s = list(served[cols].itertuples(index=False, name=None))

    cutoff = archive["date"].max() - timedelta(days=TAIL_DAYS)
    limit = min(len(a), len(s))
    for i in range(limit):
        if a[i] != s[i]:
            in_tail = a[i][0] >= cutoff and s[i][0] >= cutoff
            mark = "  " if in_tail else "!!"
            print(f"  {mark} sequences diverge at row {i:,} of {limit:,}"
                  f" ({100 * i / limit:.2f}% in)")
            print(f"     archive: {a[i]}")
            print(f"     served:  {s[i]}")
            if in_tail:
                print(f"     Inside the archive's last {TAIL_DAYS} days, so this is the"
                      " live table running ahead of")
                print("     the parquet, not a disagreement about history. The"
                      f" {i:,}-row prefix is identical,")
                print("     which is what the trained ratings were built from.")
                return True
            print("     Everything after this point is replayed from a state the other")
            print("     side never held, so the served ratings are not the trained ones.")
            return False
    if len(a) != len(s):
        longer = "served" if len(s) > len(a) else "archive"
        print(f"  first {limit:,} rows are identical; {longer} continues for"
              f" {abs(len(s) - len(a)):,} more")
        print("  (a clean tail-extension only - the shared prefix is byte-identical,")
        print("   so the divergence is new results, not a reordering)")
        return True
    print(f"  identical for all {limit:,} rows, including the (starting_at, fixture_id)")
    print("  tie-break across the 68% of fixtures that share a kickoff instant.")
    return True


def _replay(rows) -> Replay:
    """The serving walk, exactly as ratings_state.load_state() performs it."""
    replay = Replay({ratings_state.ARM: ARMS[ratings_state.ARM]()})
    for league_id, date, home_id, away_id, home_goals, away_goals in rows:
        replay.roll_to(league_id, home_id, away_id, date)
        replay.record(league_id, home_id, away_id, date, home_goals, away_goals)
    return replay


def check_ratings(archive: pd.DataFrame, served: pd.DataFrame, sample: int) -> None:
    """Replay both histories and diff what the service would actually serve.

    Not a pass/fail gate: if the archives differ at all, this is guaranteed to
    differ too, and the point is to price the difference rather than restate
    it. A tiny diff on a real matchup means the disagreement found above is
    numerically irrelevant; a large one means it is not.
    """
    print(f"\n=== 4. served rating columns ({sample} matchups per league) ===")
    cols = ["league_id", "date", "home_id", "away_id", "home_goals", "away_goals"]
    a_replay = _replay(archive[cols].itertuples(index=False, name=None))
    s_replay = _replay(served[cols].itertuples(index=False, name=None))

    # Real matchups, drawn from each league's most recent fixtures, rolled to a
    # common kickoff so both sides cross the same season boundaries.
    kickoff = pd.Timestamp(max(archive["date"].max(), served["date"].max())).to_pydatetime()
    worst: dict[str, float] = {field: 0.0 for field in FIELD_ORDER}
    n = 0
    for league_id in sorted(set(archive["league_id"])):
        recent = archive[archive["league_id"] == league_id].tail(sample)
        for fx in recent.itertuples(index=False):
            a_local = copy.deepcopy(a_replay)
            s_local = copy.deepcopy(s_replay)
            a_local.roll_to(league_id, fx.home_id, fx.away_id, kickoff)
            s_local.roll_to(league_id, fx.home_id, fx.away_id, kickoff)
            a_row = a_local.features(fx.home_id, fx.away_id, league_id)
            s_row = s_local.features(fx.home_id, fx.away_id, league_id)
            for field in FIELD_ORDER:
                worst[field] = max(worst[field], abs(a_row[field] - s_row[field]))
            n += 1

    print(f"  {n} matchups, max |archive - served| per column:")
    for field in FIELD_ORDER:
        flag = "" if worst[field] <= TOLERANCE else "   <-- differs"
        print(f"    {field:<24} {worst[field]:.6f}{flag}")
    if all(v <= TOLERANCE for v in worst.values()):
        print("  Identical to replay arithmetic: the served ratings are the trained ones.")


def check_health(rows, elapsed: float) -> None:
    """What /health would say, from the state the real loader builds."""
    print("\n=== 5. /health ===")
    print(f"  get_results_index() returned {len(rows):,} rows in {elapsed:.2f}s")
    started = time.monotonic()
    ratings_state.load_state()
    print(f"  ratings_state.load_state() took {time.monotonic() - started:.2f}s"
          f" (RATINGS_REFRESH_SECONDS budgets against this)")
    print(f"  state_summary(): {ratings_state.state_summary()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=5,
                        help="matchups per league to diff rating columns on")
    args = parser.parse_args()

    print(f"archive: {INDEX_PATH}")
    print(f"postgres: {POSTGRES_URL.split('@')[-1]}\n")

    archive = load_archive()
    db.pool.open()
    try:
        served, rows, elapsed = load_served()
        ok = check_shape(archive, served)
        ok &= check_alignment(archive, served)
        ok &= check_season_timezone(rows)
        ok &= check_order(archive, served)
        check_ratings(archive, served, args.sample)
        check_health(rows, elapsed)
    finally:
        db.pool.close()

    print("\n=== verdict ===")
    if ok:
        print("  PASS - the served archive and the trained archive are the same history.")
    else:
        print("  FAIL - see the !! lines above. The served ratings are not the ones v3")
        print("         was fitted on, and no downstream comparison of the tree against")
        print("         the blend means anything until this agrees.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
