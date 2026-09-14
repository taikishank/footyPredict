"""Summarizes real-life model performance from the graded rows
backtest/grade_calls.py writes to prediction_calls. Run after grade_calls.py;
read-only.

`python backtest/report.py` prints the all-time summary for the served model
(config's SERVED_MODEL_VERSION, currently v2). For a time-windowed view
(--today / --week) or another model version (--model v1), use
backtest/stats.py, which is the CLI wrapper around the same functions here.

Every figure is for exactly one model version: prediction_calls holds the v1
record from before the serving switch alongside the v2 record, and averaging
across the two would describe no model that ever existed.

Every pre-kickoff prediction is graded, so the summary is printed once per
segment, in two groups:

- by edge flag: all graded predictions (the honest calibration picture),
  then the flagged-edge subset and the unflagged remainder. The flagged
  segment is the one that matters for whether the edge detector is picking
  out real disagreements with the market - compare its Brier-vs-market gap
  and hit rate against the unflagged segment, which is the control group.
- by model confidence: the same low/medium/high tiers the frontend badges
  each pick with, so "does a high-confidence badge actually mean anything?"
  is answerable from the graded record.
- by double-chance confidence (single-outcome table only): the same ladder
  applied to the call's double-chance framing, 1 - model_prob against a 2/3
  baseline. The two ladders are mutually exclusive by construction - a call
  can only clear the 1/3 baseline or the 2/3 one, never both - so this ranks
  precisely the calls the model-confidence ladder pins at zero and dumps in
  its "low" bucket.

Within each segment it prints:
- hit rate on the edge side, and the model's own average confidence in that
  side (compare the two - a well-calibrated model's hit rate should track
  its average predicted probability, not just beat 50%)
- mean Brier score for flagged calls vs. the market's Brier score on the
  same fixtures - the real test of whether a flagged edge is signal rather
  than noise: is the model *more* accurate than the market where it claims
  to disagree with it?
- flat-stake paper P&L (1 unit per call), for fixtures with a recorded price
- hit rate broken out by edge-size bucket, since a model claiming a 20-point
  edge should be right far more often than one claiming a 5-point edge if
  its probabilities are meaningful

All of it is printed a second time for double_chance_calls - the same
graded calls reframed as a double-chance bet (see grade_calls.py) - so the
straight single-outcome pick and its lower-payout, easier-to-win sibling can
be compared side by side.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

# Mirrors inference-py/app/config.py's SERVED_MODEL_VERSION: report on what
# is actually being served unless asked otherwise.
DEFAULT_MODEL_VERSION = os.environ.get("SERVED_MODEL_VERSION", "v2")

_EDGE_BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.01)]

# Confidence tiers, mirroring CONFIDENCE_MEDIUM_MIN / CONFIDENCE_HIGH_MIN in
# frontend/src/app/upcoming-fixtures/upcoming-fixtures.component.ts. Keep the
# two in sync: the point of this segment is to grade the same badge the UI
# shows, so a divergence here would silently measure something else.
_CONFIDENCE_MEDIUM_MIN = 0.30
_CONFIDENCE_HIGH_MIN = 0.65

# A double chance covers 2 of 3 outcomes, so a model that knows nothing still
# says 2/3.
_DC_BASELINE = 2 / 3

# How the graded rows are sliced. `flagged` is the state of the graded
# snapshot itself; a fixture that was flagged earlier but no longer was by
# kickoff lands in the unflagged segment and is picked up by ever_flagged.
# Predicates are fixed literals, never user input - see the note below.
# Third field: whether the segment's P&L is a bet the strategy would
# actually have placed. Only flagged calls are; the others are priced at the
# same flat stake for comparison, but nobody would have staked them.
_SEGMENTS = [
    ("edge flag", "all graded predictions", "TRUE", False),
    ("edge flag", "flagged edges", "c.flagged", True),
    ("edge flag", "unflagged (control)", "NOT c.flagged", False),
    ("edge flag", "flagged at some point pre-kickoff", "c.ever_flagged", False),
]

@dataclass(frozen=True)
class CallTable:
    """One graded-call table, plus what a no-information model would say.

    Both tables share the same column shape (fixture_id, model_prob,
    market_prob, edge, correct, brier_score, market_brier_score, pnl_units,
    flagged, ever_flagged); they differ in what model_prob is a probability
    *of*, which is what `baseline` captures. A single outcome is 1 of 3, so
    a model that knows nothing still says 1/3; a double-chance bet covers 2
    of 3, so its no-information baseline is 2/3. Confidence is measured
    against that, not against 0.
    """

    label: str
    table: str
    baseline: float
    # SQL for this row's double-chance probability, or None where that would
    # just restate the row's own confidence. double_chance_calls.model_prob
    # IS the double-chance probability, so its "model confidence" group is
    # already the double-chance confidence - printing it twice would be
    # noise, not a second view.
    dc_prob_sql: str | None = None


# Names are fixed literals, never user input, so f-string interpolation of
# table names and baselines below is safe.
_CALL_TABLES = [
    # A single-outcome call's double-chance framing is the bet that excludes
    # it, so its probability is 1 - model_prob (grade_calls.py builds
    # double_chance_calls with exactly that identity).
    CallTable(
        "single-outcome pick", "prediction_calls", 1 / 3,
        dc_prob_sql="(1 - c.model_prob::float)",
    ),
    CallTable("double-chance pick", "double_chance_calls", 2 / 3),
]


def _confidence_score_sql(baseline: float, prob_sql: str = "c.model_prob::float") -> str:
    """SQL for the frontend's confidence score: a probability rescaled so that
    0 = the no-information baseline and 1 = certainty, clamped to [0, 1].

    Mirrors confidenceFor() in upcoming-fixtures.component.ts. Distinct from
    model_prob itself - a 50% call on a single outcome is a meaningful lean
    (score 0.25), while a 50% call on a double chance is worse than saying
    nothing (score 0).
    """
    return f"greatest(0.0, least(1.0, ({prob_sql} - {baseline!r}) / {1.0 - baseline!r}))"


def _confidence_tiers(group: str, score: str) -> list[tuple[str, str, str, bool]]:
    """The same low/medium/high ladder the frontend badges, over any score."""
    med, high = _CONFIDENCE_MEDIUM_MIN, _CONFIDENCE_HIGH_MIN
    return [
        (group, f"high (score >= {high:g})", f"{score} >= {high!r}", False),
        (group, f"medium ({med:g}-{high:g})", f"{score} >= {med!r} AND {score} < {high!r}", False),
        (group, f"low (< {med:g})", f"{score} < {med!r}", False),
    ]


def _segments_for(call_table: "CallTable") -> list[tuple[str, str, str, bool]]:
    """(group, label, SQL predicate, is-a-real-stake) for every segment."""
    segments = _SEGMENTS + _confidence_tiers(
        "model confidence", _confidence_score_sql(call_table.baseline)
    )
    if call_table.dc_prob_sql is not None:
        segments += _confidence_tiers(
            "double-chance confidence",
            _confidence_score_sql(_DC_BASELINE, call_table.dc_prob_sql),
        )
    return segments


@dataclass(frozen=True)
class Window:
    """A time slice of the graded rows.

    Windowed on the fixture's kickoff, not on graded_at: "the last 24 hours"
    should mean the matches that were played in it, not the rows a cron
    happened to grade in it. A fixture only becomes gradeable ~3h after
    kickoff (see grade_calls.py), so the newest results in a --today window
    are always a few hours old.
    """

    label: str
    # SQL predicate over the joined call row `c` / fixture `f`, plus the
    # params it consumes. Built here, never from user input.
    sql: str = "TRUE"
    params: tuple = ()

    @classmethod
    def all_time(cls) -> "Window":
        return cls("all time")

    @classmethod
    def last(cls, label: str, age: timedelta) -> "Window":
        return cls(label, "f.starting_at >= now() - %s", (age,))

    def for_model(self, model_version: str) -> "Window":
        """Narrows the window to one model's graded rows. The version is a
        query param, not interpolated, so an unknown value is an empty
        report rather than a SQL error."""
        return Window(
            f"{self.label}, model {model_version}",
            f"({self.sql}) AND c.model_version = %s",
            self.params + (model_version,),
        )


def _print_segment(
    cur, table: str, segment: str, where: str, staked: bool, window: Window
) -> None:
    scope = f"({where}) AND ({window.sql})"
    frm = f"FROM {table} c JOIN fixtures f USING (fixture_id)"

    cur.execute(f"SELECT count(*) {frm} WHERE {scope}", window.params)
    (total,) = cur.fetchone()
    print(f"\n-- {segment} --")
    if total == 0:
        print("no graded predictions in this segment yet")
        return

    cur.execute(
        f"""
        SELECT
            avg(c.correct::int)::float,
            avg(c.model_prob)::float,
            avg(c.brier_score)::float,
            avg(c.market_brier_score)::float,
            sum(c.pnl_units) FILTER (WHERE c.pnl_units IS NOT NULL),
            count(*) FILTER (WHERE c.pnl_units IS NOT NULL)
        {frm}
        WHERE {scope}
        """,
        window.params,
    )
    hit_rate, avg_conf, brier, market_brier, pnl, priced = cur.fetchone()

    print(f"predictions graded:   {total}")
    print(f"hit rate:             {hit_rate:.1%}  (avg model confidence: {avg_conf:.1%})")
    print(f"model brier score:    {brier:.4f}  (lower is better)")
    print(f"market brier score:   {market_brier:.4f}  (edge over market: {market_brier - brier:+.4f})")
    pnl_label = "paper P&L" if staked else "paper P&L (notional)"
    if priced:
        print(f"{pnl_label + ':':<22}{pnl:+.2f} units over {priced} priced call(s)")
    else:
        print(f"{pnl_label + ':':<22}no priced calls yet")

    print("hit rate by edge size:")
    for lo, hi in _EDGE_BUCKETS:
        cur.execute(
            f"""
            SELECT count(*), avg(c.correct::int)::float
            {frm}
            WHERE {scope} AND abs(c.edge) >= %s AND abs(c.edge) < %s
            """,
            window.params + (lo, hi),
        )
        n, bucket_hit_rate = cur.fetchone()
        label_bucket = f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">={lo:.0%}"
        if n:
            print(f"  {label_bucket:>8}: {bucket_hit_rate:.1%}  (n={n})")
        else:
            print(f"  {label_bucket:>8}: no calls yet")


def print_summary(cur, call_table: CallTable, window: Window) -> None:
    print(f"\n=== {call_table.label} ({call_table.table}) ===")
    cur.execute(
        f"SELECT count(*) FROM {call_table.table} c JOIN fixtures f USING (fixture_id) "
        f"WHERE {window.sql}",
        window.params,
    )
    (total,) = cur.fetchone()
    if total == 0:
        print(f"no graded predictions in this window ({window.label})")
        return

    group = None
    for seg_group, segment, where, staked in _segments_for(call_table):
        if seg_group != group:
            group = seg_group
            print(f"\n~~ by {group} ~~")
        _print_segment(cur, call_table.table, segment, where, staked, window)


def print_report(window: Window, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    scoped = window.for_model(model_version)
    print(f"model performance - {scoped.label} (kickoffs), all leagues")
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM prediction_calls WHERE model_version = %s",
            (model_version,),
        )
        if cur.fetchone()[0] == 0:
            print(
                f"no graded {model_version} predictions yet - run "
                f"backtest/grade_calls.py --model {model_version} first"
            )
            _print_other_versions(cur, model_version)
            return
        for call_table in _CALL_TABLES:
            print_summary(cur, call_table, scoped)


def _print_other_versions(cur, model_version: str) -> None:
    """A v2 report reading empty is expected right after the serving switch,
    but it looks identical to a broken pipeline. Naming what *is* graded
    tells the two apart at a glance."""
    cur.execute(
        "SELECT model_version, count(*) FROM prediction_calls "
        "WHERE model_version <> %s GROUP BY 1 ORDER BY 1",
        (model_version,),
    )
    rows = cur.fetchall()
    if rows:
        other = ", ".join(f"{v}: {n}" for v, n in rows)
        print(f"(graded rows under other versions - {other})")


def main() -> None:
    print_report(Window.all_time())


if __name__ == "__main__":
    main()
