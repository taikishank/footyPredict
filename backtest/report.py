"""Summarizes real-life model performance from the graded rows
backtest/grade_calls.py writes to prediction_calls. Run after grade_calls.py;
read-only.

`python backtest/report.py` prints the all-time summary for the served model
(config's SERVED_MODEL_VERSION, currently v3). For a time-windowed view
(--today / --week) or another model version (--model v2), use
backtest/stats.py, which is the CLI wrapper around the same functions here.

Every figure is for exactly one model version: prediction_calls holds the v1
and v2 records from before their respective serving switches alongside the v3
record, and averaging across them would describe no model that ever existed.

Output is tables, not stanzas. The headline is one table of six rows - each
arm's whole graded record, the subset the edge gate would actually have
staked, and the complement the gate refused - because that comparison is the
reason the report exists and it was previously spread over four screens of
prose. The refused rows are there so the gate can be judged: staked-against-
refused is the only pairing that says whether the gate selects anything, since
a staked row can look fine while being a random sample of the arm. Below it,
each arm gets one table per grouping, one row per segment, so
segment-against-segment is a column read rather than a scroll.

Every pre-kickoff prediction is graded, so each arm is sliced into segments,
in two groups:

- by edge flag: all graded predictions (the honest calibration picture),
  then the stakes subset - rows whose own selected edge clears
  EDGE_FLAG_THRESHOLD, the only ones whose P&L is a bet anyone would place -
  then the flagged-edge subset and the unflagged remainder. The flagged
  segment is the one that matters for whether the edge detector is picking
  out real disagreements with the market - compare its Brier-vs-market gap
  and hit rate against the unflagged segment, which is the control group.
  `flagged` is the snapshot's fixture-level max |edge|, which is not the
  same question as "was the side we bet worth betting" - that is the stakes
  segment. See _STAKE_SEGMENT.
- by model confidence: the same low/medium/high tiers the frontend badges
  each pick with, so "does a high-confidence badge actually mean anything?"
  is answerable from the graded record.
- by double-chance confidence (single-outcome table only): the same ladder
  applied to the call's double-chance framing, 1 - model_prob against a 2/3
  baseline. The two ladders are mutually exclusive by construction - a call
  can only clear the 1/3 baseline or the 2/3 one, never both - so this ranks
  precisely the calls the model-confidence ladder pins at zero and dumps in
  its "low" bucket.

Each segment row carries:
- hit rate as hits/n with the rate in brackets, because the denominator is
  half the number: 62.5% on 16 calls and 62.5% on 400 are not the same claim
- the model's own average confidence in the side it took (compare it with the
  hit rate - a well-calibrated model's hit rate should track its average
  predicted probability, not just beat 50%)
- mean Brier score against the market's Brier score on the same fixtures, in
  one cell and in that order - the real test of whether a flagged edge is
  signal rather than noise: is the model *more* accurate than the market
  where it claims to disagree with it? The market number has to sit beside
  it, since 0.59 is a bad Brier in the abstract and a fine one on three-way
  football.
- paper P&L in units, and ROI over units actually staked

Then one more table per arm: hit rate by edge-size bucket, segments as rows
and buckets as columns, since a model claiming a 20-point edge should be
right far more often than one claiming a 5-point edge if its probabilities
are meaningful - and that claim is about how a row moves left to right.

After both arms, an overlap section prices them as one position: 1 unit on
the single pick and 1 unit on the double chance that covers it, on the same
fixture. Each arm is otherwise graded as if it were the only bet on the book,
but they come from one snapshot and settle on one result, so anyone running
both holds 2 units per fixture with correlated downside - the single winning
implies the covering double chance winning, so the position has three states
(both legs won / DC leg only / both lost) and the combined ROI is not the
average of the two arms' ROIs. Coverage is 100% by construction for anything
graded since the arms were split, and 0% for the v1 rows graded before it,
where the DC bet was call_side's exact complement and holding both is a hedge;
the section says which regime it is looking at rather than printing a number
that means nothing.

All of it is printed a second time for double_chance_calls, and --compare
prints its head-to-head too. That arm is an independent measurement, not a
reframing: it selects its own side by max double-chance edge and settles
against a price carrying the fixture's own 1X2 margin (see grade_calls.py).
While it was derived from the singles call_side, its hit rate was exactly
1 - the singles hit rate and its price carried no margin at all, so the two
tables could not be compared on either hit rate or ROI.

The two arms have different per-bet volatility - roughly 1.05 units per bet
on double chance against ~1.5 on singles - so the comparison prints ROI and
per-bet SD together. A bare ROI difference between arms with 1.5x different
spread is not readable on its own.
"""
from __future__ import annotations

import argparse
import sys
import os
from dataclasses import dataclass
from datetime import timedelta

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

# Mirrors inference-py/app/config.py's SERVED_MODEL_VERSION: report on what is
# actually being served unless asked otherwise. The single declaration in
# backtest/ - stats.py and grade_calls.py both import this one rather than
# restating the literal.
#
# The fallback is hand-copied from config.py and has to move when a promotion
# moves that one. It is not imported from there because app/config.py requires
# DYNAMODB_TABLE_NAME and AWS_REGION at import time, and backtest/ has no
# business needing DynamoDB configured to print a table. The env var is the
# real mechanism; the literal only matters when it is unset, which is exactly
# when a stale copy grades the wrong model's rows in silence - the v2 -> v3
# promotion found this default still saying "v2" while the service wrote v3.
import selection

DEFAULT_MODEL_VERSION = os.environ.get("SERVED_MODEL_VERSION", "v3")

# Mirrors inference-py/app/config.py's SERVED_BTTS_MODEL_VERSION, hand-copied
# for the same reason DEFAULT_MODEL_VERSION is.
DEFAULT_BTTS_MODEL_VERSION = os.environ.get("SERVED_BTTS_MODEL_VERSION", "btts-v1")

# Mirrors inference-py/app/config.py's EDGE_FLAG_THRESHOLD, and hand-copied
# for the same reason DEFAULT_MODEL_VERSION is: app/config.py demands
# DynamoDB configuration at import time and a report has no business needing
# it. This is the line between "the model priced this" and "the strategy
# would have staked this", which is load-bearing for ROI - see
# _STAKE_SEGMENT below.
EDGE_FLAG_THRESHOLD = float(os.environ.get("EDGE_FLAG_THRESHOLD", "0.05"))

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

# The predicate that makes a graded row an actual stake: this arm's own
# selected edge clears the flag threshold.
#
# Both arms select by max positive edge, and that selection can never refuse:
# the model's three probabilities sum to 1 and the de-vigged market's do too,
# so the three edges sum to exactly 0 and at least one is always non-negative
# (the DC arm's three label edges sum to 0 for the same reason - both sides
# sum to 2). Measured: 0 refusals over 43 fixtures on both arms. So ROI over
# every graded row is the ROI of betting every fixture, which is not a
# strategy; only rows clearing the threshold are bets anyone would place.
#
# Note this is the ROW's edge, not `c.flagged`. `flagged` is the snapshot's
# fixture-level max |edge| >= threshold, so it can be true because of a large
# NEGATIVE edge on some third outcome that neither arm backed. It stays a
# segment below, as the calibration control it always was, but it is no
# longer what "stake" means.
_STAKE_SEGMENT = ("edge flag", f"stakes (edge >= {EDGE_FLAG_THRESHOLD:.0%})",
                  f"c.edge >= {EDGE_FLAG_THRESHOLD!r}", True)

# How the graded rows are sliced. `flagged` is the state of the graded
# snapshot itself; a fixture that was flagged earlier but no longer was by
# kickoff lands in the unflagged segment and is picked up by ever_flagged.
# Predicates are fixed literals, never user input - see the note below.
# Third field: whether the segment's P&L is a bet the strategy would
# actually have placed. Only the stake segment is; the others are priced at
# the same flat stake for comparison, but nobody would have staked them.
# Everything is still graded, flagged or not - calibration measured on a
# self-selected slice tells you nothing (see grade_calls.py).
_SEGMENTS = [
    ("edge flag", "all graded predictions", "TRUE", False),
    _STAKE_SEGMENT,
    ("edge flag", "flagged edges", "c.flagged", False),
    ("edge flag", "unflagged (control)", "NOT c.flagged", False),
    ("edge flag", "flagged at some point pre-kickoff", "c.ever_flagged", False),
]


def stake_predicate() -> str:
    """The SQL for "this row is a bet the strategy would have placed".

    Read off _SEGMENTS rather than restated, so the ROI gate in
    print_comparison() and the staked segment in the summary can never drift
    apart into two different definitions of a stake.
    """
    staked = [seg for seg in _SEGMENTS if seg[3]]
    assert len(staked) == 1, "exactly one segment is the real-stake segment"
    return staked[0][2]


def no_stake_predicate() -> str:
    """The complement: "this row was priced but the edge gate refused it".

    Negated from stake_predicate() rather than written out as `c.edge <
    threshold`, so the two can never describe overlapping or gapped sets of
    rows if the gate ever stops being a single comparison.
    """
    return f"NOT ({stake_predicate()})"

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
    # Column naming the side this arm actually bet. The two arms select
    # independently (see grade_calls.py), so "did the two versions call the
    # same thing?" has to be asked of each arm's own column.
    side_col: str = "call_side"
    # SQL for this row's double-chance probability, or None where that would
    # just restate the row's own confidence. double_chance_calls.model_prob
    # IS the double-chance probability, so its "model confidence" group is
    # already the double-chance confidence - printing it twice would be
    # noise, not a second view.
    dc_prob_sql: str | None = None


# Names are fixed literals, never user input, so f-string interpolation of
# table names and baselines below is safe.
_CALL_TABLES = [
    # The double-chance framing OF THIS CALL - the bet excluding call_side,
    # probability 1 - model_prob. This is a confidence ladder over the
    # singles arm's own picks, not a view of double_chance_calls: since the
    # arms were split, that table selects its own side and generally
    # excludes a different outcome than call_side, so the two are no longer
    # the same bet (see grade_calls.py).
    CallTable(
        "single-outcome pick", "prediction_calls", 1 / 3,
        dc_prob_sql="(1 - c.model_prob::float)",
    ),
    CallTable("double-chance pick", "double_chance_calls", 2 / 3,
              side_col="dc_label"),
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

    def for_model(self, model_version: str, call_rule: str) -> "Window":
        """Narrows the window to one model's graded rows under one selection
        rule. Both are query params, not interpolated, so an unknown value is
        an empty report rather than a SQL error.

        The rule is not optional, and that is the point: prediction_calls now
        holds one row per fixture PER RULE, so a predicate naming only the
        model silently averages max_edge and max_prob into a number that
        describes neither. Every aggregate in this file is scoped through
        here, so requiring the argument makes that impossible to forget
        rather than something each caller has to remember."""
        return Window(
            f"{self.label}, model {model_version}, rule {call_rule}",
            f"({self.sql}) AND c.model_version = %s AND c.call_rule = %s",
            self.params + (model_version, call_rule),
        )


# ---------------------------------------------------------------------------
# Rendering. Every number below is printed inside a box-drawn table whose rows
# are segments and whose columns are the four questions actually being asked
# of a graded record: did it win, did it pay, did it pay per unit staked, and
# was it better calibrated than the market. The prose format this replaced
# printed the same figures as one stanza per segment, which made the only
# comparison that matters - segment against segment, arm against arm - a
# scrolling exercise.

_H, _V = "─", "│"
_CORNERS = ("┌┬┐", "├┼┤", "└┴┘")


def _rule(widths: list[int], corners: str) -> str:
    left, mid, right = corners
    return left + mid.join(_H * (w + 2) for w in widths) + right


def _render_table(
    headers: list[str], rows: list[list[str]], aligns: str | None = None
) -> None:
    """Prints one box-drawn table. First column left-aligned, the rest right,
    unless `aligns` overrides with one of "<>^" per column.

    Cells are pre-formatted strings, not numbers: the formatting decisions
    (how many places, whether a sign is shown, what an absent value looks
    like) belong with the metric, not with the layout.
    """
    if not rows:
        return
    aligns = aligns or "<" + ">" * (len(headers) - 1)
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def line(cells: list[str], how: str) -> str:
        body = f" {_V} ".join(
            f"{c:{how[i]}{widths[i]}}" for i, c in enumerate(cells)
        )
        return f"{_V} {body} {_V}"

    print(_rule(widths, _CORNERS[0]))
    print(line(headers, "^" * len(headers)))
    for row in rows:
        print(_rule(widths, _CORNERS[1]))
        print(line(row, aligns))
    print(_rule(widths, _CORNERS[2]))


_ABSENT = "–"  # en dash: this segment has no value for this column


def _fmt_hits(s: "SegmentStats") -> str:
    """"13/43 (30.2%)" - the count is half the number. A 62.5% hit rate on 16
    calls and one on 400 are the same cell in a rate-only table."""
    return f"{s.hits}/{s.n} ({s.hit_rate:.1%})"


def _fmt_pnl(s: "SegmentStats") -> str:
    return _ABSENT if s.pnl is None else f"{s.pnl:+.2f}u"


def _fmt_roi(s: "SegmentStats") -> str:
    return _ABSENT if s.roi is None else f"{s.roi:+.1%}"


def _fmt_brier(s: "SegmentStats") -> str:
    """Model against market on the same fixtures, side by side and in that
    order, because the number alone says nothing: 0.59 is a bad Brier in the
    abstract and a good one on three-way football."""
    return f"{s.brier:.4f} vs {s.market_brier:.4f}"


def _fmt_conf(s: "SegmentStats") -> str:
    return f"{s.avg_conf:.1%}"


# ---------------------------------------------------------------------------
# Querying


@dataclass(frozen=True)
class SegmentStats:
    """One segment's graded record. `pnl`/`roi` are None when nothing in the
    segment carried a price; `roi` is P&L over units actually staked, which
    is not the same denominator as `priced` once stakes stop being flat."""

    n: int
    hits: int
    hit_rate: float
    avg_conf: float
    brier: float
    market_brier: float
    pnl: float | None
    priced: int
    roi: float | None

    @property
    def edge_over_market(self) -> float:
        return self.market_brier - self.brier


def _segment_stats(cur, table: str, where: str, window: Window) -> SegmentStats | None:
    """One round trip per segment. None when the segment is empty, which is
    normal - a confidence tier can legitimately hold no calls."""
    cur.execute(
        f"""
        SELECT count(*),
               sum(c.correct::int),
               avg(c.model_prob)::float,
               avg(c.brier_score)::float,
               avg(c.market_brier_score)::float,
               sum(c.pnl_units) FILTER (WHERE c.pnl_units IS NOT NULL),
               count(*) FILTER (WHERE c.pnl_units IS NOT NULL),
               sum(c.staked_units) FILTER (WHERE c.pnl_units IS NOT NULL)
        FROM {table} c JOIN fixtures f USING (fixture_id)
        WHERE ({where}) AND ({window.sql})
        """,
        window.params,
    )
    n, hits, avg_conf, brier, market_brier, pnl, priced, staked = cur.fetchone()
    if not n:
        return None
    return SegmentStats(
        n=n,
        hits=int(hits),
        hit_rate=int(hits) / n,
        avg_conf=float(avg_conf),
        brier=float(brier),
        market_brier=float(market_brier),
        pnl=None if pnl is None else float(pnl),
        priced=priced,
        roi=None if not pnl or not staked else float(pnl) / float(staked),
    )


def _bucket_rows(cur, table: str, where: str, window: Window) -> dict[str, tuple[int, float]]:
    """Hit rate by edge-size bucket, one query for the whole ladder rather
    than one per rung. Keyed by the bucket's printed label."""
    cases = " ".join(
        f"WHEN abs(c.edge) >= {lo!r} AND abs(c.edge) < {hi!r} THEN {i}"
        for i, (lo, hi) in enumerate(_EDGE_BUCKETS)
    )
    cur.execute(
        f"""
        SELECT CASE {cases} END AS bucket,
               count(*), avg(c.correct::int)::float
        FROM {table} c JOIN fixtures f USING (fixture_id)
        WHERE ({where}) AND ({window.sql})
        GROUP BY 1
        """,
        window.params,
    )
    out = {}
    for bucket, n, hit_rate in cur.fetchall():
        if bucket is None:
            continue
        out[_bucket_label(*_EDGE_BUCKETS[bucket])] = (n, float(hit_rate))
    return out


def _bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.0%}-{hi:.0%}" if hi < 1 else f">={lo:.0%}"


# ---------------------------------------------------------------------------
# The summary


#: The headline table's rows: each arm, whole record, staked subset and the
#: rows the gate refused. This is the one comparison the report exists to make
#: - the singles arm against the double-chance arm, and each arm's full graded
#: record against the bets the edge gate would actually have placed - so it is
#: printed first and on its own, ahead of every breakdown.
#:
#: The below-gate rows are the complement of the staked ones, not a third
#: independent slice: whole record = staked + below gate, exactly, on both
#: arms. They are here because the staked row alone cannot say whether the
#: gate is doing anything. A gate that keeps the arm's average performance
#: unchanged is selecting at random, however good the staked row looks in
#: isolation; the comparison that answers that is staked against refused, and
#: it has to be a column read in the headline table rather than two rows
#: buried in different per-arm breakdowns.
_HEADLINE_ROWS = [
    ("singles", _CALL_TABLES[0], "TRUE"),
    ("double chance", _CALL_TABLES[1], "TRUE"),
    (f"singles, {_STAKE_SEGMENT[1]}", _CALL_TABLES[0], _STAKE_SEGMENT[2]),
    (f"double, {_STAKE_SEGMENT[1]}", _CALL_TABLES[1], _STAKE_SEGMENT[2]),
    (f"singles, below gate (edge < {EDGE_FLAG_THRESHOLD:.0%})",
     _CALL_TABLES[0], no_stake_predicate()),
    (f"double, below gate (edge < {EDGE_FLAG_THRESHOLD:.0%})",
     _CALL_TABLES[1], no_stake_predicate()),
]


def print_headline(cur, window: Window) -> bool:
    """Both arms, whole record and staked subset, in one table. Returns False
    when nothing is graded in the window at all."""
    rows = []
    for label, call_table, where in _HEADLINE_ROWS:
        stats = _segment_stats(cur, call_table.table, where, window)
        if stats is None:
            continue
        rows.append([
            label,
            _fmt_hits(stats),
            _fmt_pnl(stats),
            _fmt_roi(stats),
            _fmt_brier(stats),
        ])
    if not rows:
        return False
    _render_table(["arm", "hit rate", "P&L", "ROI", "Brier vs market"], rows)
    print(
        "P&L and ROI on the unstaked rows are notional - both arms select by\n"
        "max positive edge and that selection never refuses, so the whole-record\n"
        "rows price every fixture, which is not a strategy, and the below-gate\n"
        "rows price the bets the gate explicitly declined. The staked rows are\n"
        "the bets the edge gate would have placed; staked + below gate is the\n"
        "whole record, so read those two against each other to see whether the\n"
        "gate is selecting anything the arm's average does not already have."
    )
    return True


def print_summary(cur, call_table: CallTable, window: Window) -> None:
    """One arm's breakdown: a table per grouping, then the edge-size ladder."""
    print(f"\n=== {call_table.label} ({call_table.table}) ===")
    whole = _segment_stats(cur, call_table.table, "TRUE", window)
    if whole is None:
        print(f"no graded predictions in this window ({window.label})")
        return

    headers = ["segment", "n", "hit rate", "avg conf", "Brier vs market", "P&L", "ROI"]
    group = None
    rows: list[list[str]] = []
    for seg_group, segment, where, _staked in _segments_for(call_table):
        if seg_group != group:
            if rows:
                _render_table(headers, rows)
            rows = []
            group = seg_group
            print(f"\nby {group}:")
        stats = _segment_stats(cur, call_table.table, where, window)
        if stats is None:
            rows.append([segment, "0"] + [_ABSENT] * (len(headers) - 2))
            continue
        rows.append([
            segment,
            str(stats.n),
            _fmt_hits(stats),
            _fmt_conf(stats),
            _fmt_brier(stats),
            _fmt_pnl(stats),
            _fmt_roi(stats),
        ])
    if rows:
        _render_table(headers, rows)

    _print_edge_ladder(cur, call_table, window)


def _print_edge_ladder(cur, call_table: CallTable, window: Window) -> None:
    """Hit rate by edge size, one row per edge-flag segment.

    One table rather than a ladder repeated inside every segment stanza: the
    claim being tested is that a bigger claimed edge wins more often, which
    is a question about how a row moves left to right, and that is
    unreadable when the rungs are scattered across five stanzas. Buckets are
    the columns because there are always five of them and segment labels are
    long.
    """
    segments = [(label, where) for _g, label, where, _s in _SEGMENTS]
    by_segment = {
        label: _bucket_rows(cur, call_table.table, where, window)
        for label, where in segments
    }
    buckets = [_bucket_label(lo, hi) for lo, hi in _EDGE_BUCKETS]
    # A bucket nothing landed in is a column of dashes; drop it rather than
    # pay a column's width to say so five times.
    buckets = [b for b in buckets if any(b in found for found in by_segment.values())]

    rows = []
    for label, _where in segments:
        found = by_segment[label]
        rows.append([label] + [
            f"{found[b][1]:.1%} (n={found[b][0]})" if b in found else _ABSENT
            for b in buckets
        ])
    print("\nhit rate by edge size:")
    _render_table(["segment"] + buckets, rows)


# ---------------------------------------------------------------------------
# The overlap arm: both legs on the same fixture


def _rebind(sql: str, alias: str) -> str:
    """Rebinds a predicate written against the generic call alias `c` onto
    another alias. Lets the stake gate and the window be declared once and
    reused in the self-join and the two-table join, instead of being restated
    per query and drifting."""
    return sql.replace("c.", f"{alias}.")


#: The overlap position's segments, by which legs cleared the edge gate.
#: `stake_predicate()` is rebound onto both tables rather than restated, so
#: "would the strategy have placed this" means the same thing here as in the
#: per-arm summary, and the neither-leg row negates that same predicate so the
#: two cannot drift into an overlapping or gapped pair.
#:
#: The rows are not a partition: "either" and "neither" are complements and do
#: split the covered fixtures in two, while "both" and "single below gate, DC
#: staked" are both subsets of "either" (and are themselves disjoint - one
#: needs the single leg staked, the other needs it unstaked). The neither-leg
#: row is the control for the whole position - it is the fixtures where the
#: gate declined 2 units, so if it pays like the both-legs row, the gate is
#: choosing which fixtures to skip at random. The new row isolates the case
#: the gate is actually meant to catch: the single-outcome pick too thin on
#: its own to clear the threshold, but the double chance covering it clears
#: it anyway - if that row's ROI beats "both", the DC leg is picking up real
#: signal the single-outcome gate is discarding.
def _overlap_segments() -> list[tuple[str, str]]:
    stake = stake_predicate()
    p, d = _rebind(stake, "p"), _rebind(stake, "d")
    no_p, no_d = _rebind(no_stake_predicate(), "p"), _rebind(no_stake_predicate(), "d")
    gate = f"{EDGE_FLAG_THRESHOLD:.0%}"
    return [
        ("all covered fixtures", "TRUE"),
        (f"either leg staked (edge >= {gate})", f"({p}) OR ({d})"),
        (f"both legs staked (edge >= {gate})", f"({p}) AND ({d})"),
        (f"single below gate, DC staked (edge < {gate} / >= {gate})",
         f"({no_p}) AND ({d})"),
        (f"neither leg staked (edge < {gate})", f"({no_p}) AND ({no_d})"),
    ]


@dataclass(frozen=True)
class OverlapStats:
    """One segment of the combined position: 1 unit on the single pick and 1
    unit on the double chance that covers it, on the same fixture."""

    n: int
    both_won: int
    dc_only: int
    both_lost: int
    pnl: float | None
    priced: int
    staked: float


#: The join that defines the arm. The single pick is covered when the double
#: chance does not exclude it - "away win" as the single, "away or home" /
#: "away or draw" as the double chance. Written as the exclusion rather than
#: as a membership test because dc_label is a string ("1X") while
#: excludes_outcome names the outcome directly, so this needs no decoding.
_COVERED = "p.call_side <> d.excludes_outcome"

_OVERLAP_FROM = """
    FROM prediction_calls p
    JOIN double_chance_calls d USING (fixture_id, model_version, call_rule)
    JOIN fixtures f USING (fixture_id)
"""


def _overlap_stats(cur, where: str, window: Window) -> OverlapStats | None:
    priced = "p.pnl_units IS NOT NULL AND d.pnl_units IS NOT NULL"
    cur.execute(
        f"""
        SELECT count(*),
               count(*) FILTER (WHERE p.correct),
               count(*) FILTER (WHERE d.correct AND NOT p.correct),
               count(*) FILTER (WHERE NOT d.correct),
               sum(p.pnl_units + d.pnl_units) FILTER (WHERE {priced}),
               count(*) FILTER (WHERE {priced}),
               sum(p.staked_units + d.staked_units) FILTER (WHERE {priced})
        {_OVERLAP_FROM}
        WHERE {_COVERED} AND ({where}) AND ({_rebind(window.sql, "p")})
        """,
        window.params,
    )
    n, both_won, dc_only, both_lost, pnl, priced_n, staked = cur.fetchone()
    if not n:
        return None
    return OverlapStats(
        n=n,
        both_won=both_won,
        dc_only=dc_only,
        both_lost=both_lost,
        pnl=None if pnl is None else float(pnl),
        priced=priced_n,
        staked=0.0 if staked is None else float(staked),
    )


def _overlap_coverage(cur, window: Window) -> tuple[int, int]:
    """(covered, total) fixtures graded in both arms."""
    cur.execute(
        f"""
        SELECT count(*) FILTER (WHERE {_COVERED}), count(*)
        {_OVERLAP_FROM}
        WHERE {_rebind(window.sql, "p")}
        """,
        window.params,
    )
    return cur.fetchone()


def print_overlap(cur, window: Window) -> None:
    """The two arms held together on the same fixture.

    Each arm is graded as if it were the only bet on the book, but they are
    priced off one snapshot and settled on one result, so anyone running both
    holds a single position per fixture, not two independent ones. This
    section prices that position: 1 unit on the single pick and 1 unit on the
    double chance covering it, 2 units at risk per fixture.

    The two legs are not independent. Where the double chance covers the
    single pick, the single winning implies the double chance winning, so the
    position has exactly three states and no fourth:

      both legs won   the called outcome hit; both tickets pay
      DC leg only     one of the other two covered outcomes hit; the DC
                      ticket pays and the single is lost
      both lost       the excluded outcome hit; both tickets are lost

    That structure is why the combined ROI is not the average of the two
    arms' ROIs: the losses are perfectly correlated on the downside and only
    the upside is split, which is the whole reason to look at the pair rather
    than at two separate tables.
    """
    covered, total = _overlap_coverage(cur, window)
    print("\n=== overlap (single pick + the double chance covering it) ===")
    if total == 0:
        print(f"no fixtures graded in both arms in this window ({window.label})")
        return

    print(f"the double chance covers the single pick on {covered}/{total} "
          f"fixtures ({covered / total:.0%})")
    if covered == 0:
        _print_no_coverage()
        return
    if covered == total:
        # Not a property of this data: the singles arm takes the max signed
        # edge and the DC arm excludes the min signed edge, and the argmax
        # and argmin of three numbers differ unless all three are equal. So
        # coverage is 100% for anything graded since the arms were split, and
        # the interesting number is never the coverage rate - it is what the
        # combined position pays.
        print("which is 100% by construction: the single pick is the model's "
              "max-signed-edge\noutcome and the double chance excludes its "
              "min-signed-edge outcome, and those\ncan only coincide if all "
              "three edges are equal. See grade_calls.py.")

    rows = []
    for label, where in _overlap_segments():
        s = _overlap_stats(cur, where, window)
        if s is None:
            continue
        roi = s.pnl / s.staked if s.pnl is not None and s.staked else None
        rows.append([
            label,
            str(s.n),
            f"{s.both_won} ({s.both_won / s.n:.1%})",
            f"{s.dc_only} ({s.dc_only / s.n:.1%})",
            f"{s.both_lost} ({s.both_lost / s.n:.1%})",
            _ABSENT if s.pnl is None else f"{s.pnl:+.2f}u",
            _ABSENT if roi is None else f"{roi:+.1%}",
        ])
    print()
    _render_table(
        ["segment", "n", "both legs won", "DC leg only", "both lost", "P&L", "ROI"],
        rows,
    )
    print(
        "1 unit on each leg, so 2 units at risk per fixture and ROI is over "
        "that\nfull stake. P&L on the unstaked rows is notional, as everywhere "
        "else - the\nselection never refuses, so the first row bets every "
        "fixture and the last\nprices the 2 units the gate declined. Either "
        "and neither split the covered\nfixtures in two; both and single "
        "below gate/DC staked are disjoint subsets\nof either. That last row "
        "is the one worth reading on its own: it prices the\nfixtures where "
        "the single-outcome edge alone did not clear the gate but the\n"
        "double chance covering it did."
    )


def _print_no_coverage() -> None:
    """Zero coverage is not a bug either: it is the signature of the v1 rows,
    graded before the arms were split, when the DC bet was defined as the pair
    of outcomes EXCLUDING call_side. Those two legs are a hedge, not a
    doubled-up position - exactly one of them wins on every fixture - so
    pricing them together measures the book's margin, not the model."""
    print(
        "\nOn these rows the double chance is the exact complement of the "
        "single pick,\nwhich is how the arms were defined before the split "
        "(see grade_calls.py):\nthe DC bet excluded call_side rather than "
        "selecting its own side. Holding\nboth is a hedge - exactly one leg "
        "wins on every fixture - so the combined\nP&L measures the book's "
        "overround and nothing about the model. Nothing to\nreport here; "
        "the overlap arm only means something for v2 and later."
    )


# ---------------------------------------------------------------------------
# Rule overlap. The sections above answer "how did this rule do"; this one
# answers "what do the two rules do where they AGREE" - the slice of the book
# on which the call_rule axis is a no-op.
#
# Both rules read the same edges snapshot, so where they name the same side
# they are not merely similar bets, they are the SAME bet: same probability,
# same market probability, same edge, same price, same result. That is why
# this section has one row per segment rather than one per rule - printing
# max_edge and max_prob separately here would print each number twice.
#
# The segments are the edge gate, which is the only thing still free to vary
# once the side is fixed: agreeing fixtures whose shared edge cleared the
# threshold, and agreeing fixtures whose did not.


def _agreement_predicate(call_table: CallTable, model_version: str) -> str:
    """SQL for "the other rule called this same side", over the `c` alias
    that _segment_stats() binds.

    Written as EXISTS rather than a join so the whole section can reuse
    _segment_stats() and the formatters the other tables use, instead of
    growing a parallel query that computes the same columns slightly
    differently. model_version and the rule names are internal literals, and
    the value that varies is quoted through %r the same way the segment
    predicates above are.
    """
    side = call_table.side_col
    return (
        f"EXISTS (SELECT 1 FROM {call_table.table} o "
        f"WHERE o.fixture_id = c.fixture_id "
        f"AND o.model_version = {model_version!r} "
        f"AND o.call_rule = 'max_prob' "
        f"AND o.{side} = c.{side})"
    )


def _rule_overlap_segments(agree: str) -> list[tuple[str, str]]:
    """The two rows, both inside the agreement slice."""
    gate = f"{EDGE_FLAG_THRESHOLD:.0%}"
    return [
        (f"agreed, staked (edge >= {gate})", f"({agree}) AND ({stake_predicate()})"),
        (f"agreed, below gate (edge < {gate})",
         f"({agree}) AND ({no_stake_predicate()})"),
    ]


def _agreement_coverage(cur, call_table: CallTable, agree: str, window: Window) -> tuple[int, int]:
    """(agreed, total) fixtures graded under the max_edge rule."""
    cur.execute(
        f"""
        SELECT count(*) FILTER (WHERE {agree}), count(*)
        FROM {call_table.table} c JOIN fixtures f USING (fixture_id)
        WHERE ({window.sql})
        """,
        window.params,
    )
    return cur.fetchone()


def print_rule_overlap(cur, model_version: str, window: Window) -> None:
    """Where max_edge and max_prob call the same side, for both arms.

    The agreement rate is the headline, not a footnote. Everything the
    call_rule axis can be worth lives in the fixtures where the rules
    disagree; on the ones where they agree there is nothing to choose
    between, and a difference read off the whole book without knowing how
    much of it is shared is a difference attributed to the wrong thing.
    """
    print(f"\n=== rule overlap (max_edge and max_prob agreeing, "
          f"model {model_version}) ===")
    headers = ["segment", "n", "hit rate", "avg conf", "Brier vs market", "P&L", "ROI"]
    for call_table in _CALL_TABLES:
        scoped = window.for_model(model_version, "max_edge")
        agree = _agreement_predicate(call_table, model_version)
        agreed, total = _agreement_coverage(cur, call_table, agree, scoped)
        print(f"\n--- {call_table.label} ({call_table.table}) ---")
        if not total:
            print(f"no graded predictions in this window ({scoped.label})")
            continue
        print(f"the two rules called the same side on {agreed}/{total} "
              f"fixtures ({agreed / total:.0%})")
        if not agreed:
            continue

        rows: list[list[str]] = []
        for label, where in _rule_overlap_segments(agree):
            stats = _segment_stats(cur, call_table.table, where, scoped)
            if stats is None:
                rows.append([label, "0"] + [_ABSENT] * (len(headers) - 2))
                continue
            rows.append([
                label,
                str(stats.n),
                _fmt_hits(stats),
                _fmt_conf(stats),
                _fmt_brier(stats),
                _fmt_pnl(stats),
                _fmt_roi(stats),
            ])
        print()
        _render_table(headers, rows)

    print(
        "\nOne row per segment, not one per rule: on an agreeing fixture the\n"
        "two rules produce identical graded rows - same side off the same\n"
        "snapshot - so there is nothing to compare, and the gate is the only\n"
        "thing left that varies. The two rows are the whole agreement slice,\n"
        "split by whether that shared edge cleared the threshold. P&L on the\n"
        "below-gate row is notional, as everywhere else: those are the bets\n"
        "the gate declined."
    )


def print_report(
    window: Window,
    model_version: str = DEFAULT_MODEL_VERSION,
    call_rule: str = selection.DEFAULT_RULE,
) -> None:
    scoped = window.for_model(model_version, call_rule)
    print(f"model performance - {scoped.label} (kickoffs), all leagues")
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM prediction_calls "
            "WHERE model_version = %s AND call_rule = %s",
            (model_version, call_rule),
        )
        if cur.fetchone()[0] == 0:
            print(
                f"no graded {model_version} predictions under rule "
                f"{call_rule} yet - run backtest/grade_calls.py "
                f"--model {model_version} --rule {call_rule} first"
            )
            _print_other_versions(cur, model_version)
            return
        if not print_headline(cur, scoped):
            print(f"no graded predictions in this window ({scoped.label})")
            return
        for call_table in _CALL_TABLES:
            print_summary(cur, call_table, scoped)
        print_overlap(cur, scoped)


def print_full_report(
    window: Window,
    model_version: str = DEFAULT_MODEL_VERSION,
    rules: tuple[str, ...] = selection.RULES,
) -> None:
    """Every rule's section in turn, then the rule-overlap section.

    The sections are appended, not merged: each one is the same report
    print_report() has always produced, scoped to one rule and labelled with
    which. Nothing inside a section changes when a second rule is added -
    the figures in the max_edge section are the figures that section printed
    before call_rule existed, because the rows behind them are the same rows
    (grade_calls.py labelled every pre-existing row max_edge, which is what
    they were).
    """
    for i, call_rule in enumerate(rules):
        if i:
            print("\n" + "=" * 72)
        print_report(window, model_version, call_rule)
    if len(rules) > 1:
        print("\n" + "=" * 72)
        with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
            print_rule_overlap(cur, model_version, window)


def _print_other_versions(cur, model_version: str) -> None:
    """A v2 report reading empty is expected right after the serving switch,
    but it looks identical to a broken pipeline. Naming what *is* graded
    tells the two apart at a glance."""
    cur.execute(
        "SELECT model_version || \'/\' || call_rule, count(*) "
        "FROM prediction_calls WHERE model_version <> %s GROUP BY 1 ORDER BY 1",
        (model_version,),
    )
    rows = cur.fetchall()
    if rows:
        other = ", ".join(f"{v}: {n}" for v, n in rows)
        print(f"(graded rows under other versions - {other})")


#: Bootstrap resamples for the head-to-head interval. Matches
#: ml/rating_ablate.py's BOOTSTRAP_SAMPLES so an offline ablation and a live
#: head-to-head are read on the same scale.
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


def _paired_bootstrap(baseline: list[float], arm: list[float]) -> dict[str, float]:
    """Positive `improvement` favours the arm. The twin of
    ml/rating_ablate.paired_bootstrap, kept to the same sign convention and
    resample count so an offline margin and a graded one are comparable.

    Paired, not two-sample: both models scored the *same* fixtures, and their
    errors are highly correlated because they share 56 of 63 columns and read
    the same market. Resampling them independently would inflate the interval
    to the point of never separating anything.
    """
    import numpy as np

    diff = np.array(baseline, dtype=float) - np.array(arm, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "improvement": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_better": float((means > 0).mean()),
    }


@dataclass(frozen=True)
class ArmRow:
    """One model version's graded row for one fixture, in one arm."""

    correct: bool
    brier: float
    market_brier: float
    pnl: float | None
    side: str
    edge: float
    staked_units: float
    is_stake: bool


@dataclass(frozen=True)
class Axis:
    """Which column a head-to-head varies, and which it holds fixed.

    There are two things a graded row can differ on, and conflating them was
    the reason this file could not answer "is the draw problem the model or
    the selection rule". `model_version` names the artifact that produced the
    probabilities; `call_rule` names what backtest/selection.py did with them.
    Pinning one and pairing on the other is what makes the answer causal.

    `column` and `fixed_column` are internal constants, never user input -
    they are interpolated into SQL, while the VALUES compared always travel
    as query params.
    """

    column: str
    fixed_column: str
    noun: str


#: Two models, one rule - the original comparison. Pairing is partial: a
#: version only has rows for fixtures it served, so an intersection exists
#: only after a reconstruction like the `v2-backfill` tier, or during a
#: shadow-scoring period. print_comparison() says so rather than printing an
#: empty table.
BY_MODEL = Axis("model_version", "call_rule", "model")

#: Two rules, one model - the selection comparison. Pairing is TOTAL here,
#: which is what makes it the stronger of the two: grade_calls.py grades
#: every rule off the same edges snapshot, so both arms see byte-identical
#: probabilities on every fixture. Nothing is left for a difference to be
#: except the rule. The model-axis comparison can never be this clean,
#: because two models by definition disagree about the probabilities.
BY_RULE = Axis("call_rule", "model_version", "rule")


def _common_calls(
    cur,
    window: Window,
    baseline: str,
    arm: str,
    call_table: "CallTable",
    axis: Axis,
    fixed: str,
) -> list[tuple]:
    """Graded rows for fixtures *both* arms called, one row per fixture.

    The join is the whole point. On the model axis, each version's record
    covers the period it was serving, so `--model v2` and `--model v3`
    describe different calendars: v2's September against v3's October is a
    comparison of months, and whatever it shows would survive swapping the
    models. Restricting to fixtures both priced removes the calendar, the
    league mix and the fixture difficulty in one step.

    The other axis is held fixed rather than ignored. Without that, a
    two-model join would match each of the baseline's rules against each of
    the arm's and return four rows per fixture, silently inflating n and
    double-counting every P&L.

    Each row carries the arm's is-a-real-stake flag, evaluated in SQL from
    stake_predicate() with the aliases rebound, so the ROI gate here is
    literally the same predicate the staked segment of the summary uses.
    """
    stake = stake_predicate()
    a_stake, b_stake = _rebind(stake, "a"), _rebind(stake, "b")
    side = call_table.side_col
    cur.execute(
        f"""
        SELECT f.starting_at,
               a.correct, a.brier_score, a.market_brier_score, a.pnl_units,
               a.{side}, a.edge, a.staked_units, ({a_stake}),
               b.correct, b.brier_score, b.market_brier_score, b.pnl_units,
               b.{side}, b.edge, b.staked_units, ({b_stake})
        FROM {call_table.table} a
        JOIN {call_table.table} b
          ON b.fixture_id = a.fixture_id
         AND b.{axis.column} = %s
         AND b.{axis.fixed_column} = a.{axis.fixed_column}
        JOIN fixtures f ON f.fixture_id = a.fixture_id
        WHERE a.{axis.column} = %s
          AND a.{axis.fixed_column} = %s
          AND ({window.sql})
        ORDER BY f.starting_at
        """,
        (arm, baseline, fixed) + window.params,
    )
    return cur.fetchall()


def _arm_rows(rows: list[tuple], offset: int) -> list[ArmRow]:
    """One side of the paired join, unpacked. Baseline starts at 1, arm at 9."""
    return [
        ArmRow(
            correct=bool(r[offset]),
            brier=float(r[offset + 1]),
            market_brier=float(r[offset + 2]),
            pnl=None if r[offset + 3] is None else float(r[offset + 3]),
            side=r[offset + 4],
            edge=float(r[offset + 5]),
            staked_units=float(r[offset + 6]),
            is_stake=bool(r[offset + 7]),
        )
        for r in rows
    ]


def _roi(arm_rows: list[ArmRow]) -> tuple[float | None, float | None, int]:
    """(ROI as a fraction, per-bet stddev in units, n) over real stakes only.

    Gated on is_stake, not on every graded row: both arms select by max
    positive edge and that selection can never refuse (the three edges sum
    to 0, so one is always non-negative), so an ungated ROI is the ROI of
    betting every fixture in the fixture list. See _STAKE_SEGMENT.

    The stddev is per bet, not of the ROI, and it is what makes two ROIs
    comparable: the arms have genuinely different volatility (measured on
    the v2/v3 backfill, ~1.05 units/bet for double chance against
    ~1.45-1.64 for singles), so the DC arm converges on its true ROI
    roughly 2.4x faster in bets. Two ROI numbers without it invite reading
    a difference that the wider arm's noise would swallow.
    """
    import numpy as np

    pnl = [r.pnl for r in arm_rows if r.is_stake and r.pnl is not None]
    staked = [r.staked_units for r in arm_rows if r.is_stake and r.pnl is not None]
    if not pnl or sum(staked) == 0:
        return None, None, len(pnl)
    roi = sum(pnl) / sum(staked)
    sd = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else None
    return roi, sd, len(pnl)


def _print_arm_head_to_head(
    cur,
    call_table: "CallTable",
    window: Window,
    baseline: str,
    arm: str,
    axis: Axis,
    fixed: str,
) -> int:
    """One arm's head-to-head table. Returns the number of paired fixtures."""
    rows = _common_calls(cur, window, baseline, arm, call_table, axis, fixed)
    if not rows:
        return 0

    n = len(rows)
    base = _arm_rows(rows, 1)
    armr = _arm_rows(rows, 9)
    market_brier = [r.market_brier for r in base]
    disagreed = sum(1 for x, y in zip(base, armr) if x.side != y.side)

    print(f"\n=== {call_table.label} ({call_table.table}) ===")
    print(f"{n} fixtures graded under both, "
          f"{rows[0][0]:%Y-%m-%d} .. {rows[-1][0]:%Y-%m-%d}")
    print(f"the two {axis.noun}s called a different side on {disagreed} of them "
          f"({disagreed / n:.0%})")

    print()
    table_rows = []
    for label, side_rows in ((baseline, base), (arm, armr)):
        brier = sum(r.brier for r in side_rows) / n
        hits = sum(1 for r in side_rows if r.correct)
        pnl = sum(r.pnl for r in side_rows if r.is_stake and r.pnl is not None)
        roi, sd, stakes = _roi(side_rows)
        table_rows.append([
            label,
            f"{brier:.5f}",
            f"{hits}/{n} ({hits / n:.1%})",
            f"{pnl:+.2f}u",
            f"{roi:+.1%}" if roi is not None else _ABSENT,
            f"{sd:.2f}" if sd is not None else _ABSENT,
            str(stakes),
        ])
    # The market is scored on the same fixtures but places no bets, so every
    # column but Brier is empty for it by construction, not for want of data.
    table_rows.append(
        ["market", f"{sum(market_brier) / n:.5f}"] + [_ABSENT] * 5
    )
    _render_table(
        [axis.noun, "Brier", "hit rate", "P&L", "ROI", "SD/bet", "stakes"],
        table_rows,
    )
    print("P&L, ROI and SD/bet cover only rows clearing the "
          f"{EDGE_FLAG_THRESHOLD:.0%} edge gate - the\nbets the strategy "
          "would have placed. Brier and hit rate cover all\n"
          f"{n} graded fixtures, which is the calibration question, not the "
          "staking one.")

    base_brier = [r.brier for r in base]
    arm_brier = [r.brier for r in armr]
    if base_brier == arm_brier:
        # Not a null result - a metric that cannot move on this axis.
        # prediction_calls.brier_score scores the model's FULL three-outcome
        # distribution (grade_calls.py's _brier_score over model_probs), which
        # is a property of the snapshot, not of the side a rule picked from
        # it. Two rules reading one snapshot therefore always tie exactly, and
        # the bootstrap over that tie prints +0.00000 with P 0.000, which
        # reads like a broken measurement rather than an identity.
        #
        # The DC arm does NOT hit this: its model_prob is 1 - model[excluded],
        # so it depends on which outcome the rule excluded, and its Brier is
        # a real comparison.
        print(f"\nBrier is identical by construction here: it scores the whole "
              f"three-outcome\ndistribution, which both {axis.noun}s read from "
              f"the same snapshot. Only the\npick differs, so hit rate and ROI "
              f"are the columns that separate them.")
        return n

    stats = _paired_bootstrap(base_brier, arm_brier)
    ci = f"[{stats['ci_low']:+.5f}, {stats['ci_high']:+.5f}]"
    print(f"\n{arm} Brier gain over {baseline}: {stats['improvement']:+.5f}  "
          f"95% CI {ci}  P({arm} better) {stats['p_better']:.3f}")
    return n


def print_comparison(
    window: Window,
    baseline: str,
    arm: str,
    axis: Axis = BY_MODEL,
    fixed: str = selection.DEFAULT_RULE,
) -> None:
    """Two models, or two selection rules, head to head on both betting arms
    over the fixtures both of them called.

    Brier is the headline rather than hit rate, for the reason grade_calls.py
    records both: hit rate scores only the called side and is insensitive to
    how wrong the other two outcomes were, while Brier scores the whole
    distribution. A model can improve its hit rate by getting more confident
    on calls it already won, which is not what a promotion is trying to buy.

    Both arms are printed because they are now independent measurements.
    While double_chance_calls was derived from call_side, its hit rate was
    exactly 1 - the singles hit rate and printing it added no information;
    since the split (see grade_calls.py) the two arms select different sides
    and can disagree about which version is better, which is the whole point
    of running them side by side.
    """
    if baseline == arm:
        # The self-join succeeds and reports a perfect zero, which is a correct
        # answer to a question nobody meant to ask. Say so instead.
        print(f"{baseline} against itself is zero by construction - "
              f"pass two different {axis.noun}s.")
        return

    held = f"{axis.fixed_column.replace('_', ' ')} {fixed}"
    print(f"head to head: {baseline} vs {arm} ({axis.noun}), holding {held} - "
          f"{window.label} (kickoffs), all leagues")
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        n = 0
        for call_table in _CALL_TABLES:
            n = max(n, _print_arm_head_to_head(
                cur, call_table, window, baseline, arm, axis, fixed))

        if n == 0:
            _print_no_overlap(cur, baseline, arm, axis, fixed)
            return

        print("\nLower Brier is better, so a positive gain favours "
              f"{arm}. An interval spanning\nzero is a null result regardless "
              "of where the point estimate sits.")

        if n < 100:
            print(f"\n{n} fixtures is thin - the interval above is wide enough that "
                  "it will\nseparate almost nothing. Read it as a sanity check on "
                  "the serving path,\nnot as evidence for or against a promotion.")
            print(
                "The ROI column is thinner still. At the per-bet volatility these\n"
                "arms actually show (~1.05 units on double chance, ~1.5 on singles),\n"
                f"the standard error on a {n}-bet ROI is around +/-16%, so every\n"
                "number in that column is consistent with zero. Narrowing it to a\n"
                "+/-3% band needs roughly 5,000 bets on the double-chance arm and\n"
                "~10,000 on singles - about a year and two-plus years at the current\n"
                "grading rate. Nothing here is evidence yet; what changed is that the\n"
                "measurement is now capable of producing some."
            )


def _print_no_overlap(cur, baseline: str, arm: str, axis: Axis, fixed: str) -> None:
    """No fixture graded under both arms - say what does have rows and how to
    create an overlap, rather than printing an empty table."""
    print(f"\nno fixture has been graded under both {baseline} and {arm} "
          f"with {axis.fixed_column} = {fixed}.")
    cur.execute(
        "SELECT model_version || \'/\' || call_rule, count(*), "
        "       min(f.starting_at)::date, max(f.starting_at)::date "
        "FROM prediction_calls c JOIN fixtures f USING (fixture_id) "
        "GROUP BY 1 ORDER BY 1"
    )
    graded = cur.fetchall()
    if graded:
        print("\ngraded rows by model/rule (kickoff range):")
        for version, n, first, last in graded:
            print(f"  {version:<22} {n:>5}  {first} .. {last}")
    if axis is BY_RULE:
        # Rules pair totally when both have been graded, so an empty result
        # means one simply has not been run - not that an overlap has to be
        # manufactured the way it does on the model axis.
        print(
            f"\nEvery rule grades the same snapshots, so these two overlap\n"
            f"completely once both have run. Grade the missing one:\n"
            f"  ./inference-py/bin/python backtest/grade_calls.py "
            f"--model {fixed} --rule {arm}"
        )
        return
    print(
        "\nEach version only has rows for fixtures it was serving, so an\n"
        "intersection needs two versions to have priced the same matches.\n"
        "Either wait for a shadow-scoring period, or reconstruct the older\n"
        "version's edges over fixtures the newer one called - the same move\n"
        "that produced the `v2-backfill` tier. Until then the honest\n"
        "comparison is the offline one:\n"
        "  ./inference-py/bin/python inference-py/scripts/compare_v2_v3_served.py"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graded-performance report for one model/rule, or a "
        "head-to-head between two models or two selection rules.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_VERSION,
        help=f"model version to report on (default: {DEFAULT_MODEL_VERSION})",
    )
    parser.add_argument(
        "--rule",
        default=selection.DEFAULT_RULE,
        choices=selection.RULES,
        help=f"selection rule to report on (default: {selection.DEFAULT_RULE})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--compare-models",
        nargs=2,
        metavar=("BASELINE", "ARM"),
        help="two model versions head to head, both under --rule",
    )
    group.add_argument(
        "--compare-rules",
        nargs=2,
        metavar=("BASELINE", "ARM"),
        help="two selection rules head to head, both under --model. This is "
        "the cleaner of the two comparisons: both arms read the same edges "
        "snapshots, so the probabilities are identical and only the pick "
        "differs.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    window = Window.all_time()
    rule_given = "--rule" in sys.argv
    if args.compare_models:
        baseline, arm = args.compare_models
        print_comparison(window, baseline, arm, BY_MODEL, args.rule)
    elif args.compare_rules:
        baseline, arm = args.compare_rules
        for name in (baseline, arm):
            if name not in selection.RULES:
                raise SystemExit(
                    f"unknown rule {name!r} - known rules: "
                    f"{', '.join(selection.RULES)}"
                )
        print_comparison(window, baseline, arm, BY_RULE, args.model)
    elif rule_given:
        # An explicit --rule means "just this one", which is what the flag
        # meant before there was a second rule to print.
        print_report(window, args.model, args.rule)
    else:
        print_full_report(window, args.model)


if __name__ == "__main__":
    main()
