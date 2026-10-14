"""Summarizes real-life BTTS performance from the graded rows
backtest/grade_calls.py --market btts writes to btts_calls. Run after
grade_calls.py; read-only.

report.py's twin, deliberately not a third arm folded into it. The two
reasons are the same reason: btts_calls' model_version values (btts-v1,
btts-backfill, ...) are a different namespace from edges/prediction_calls'
(v1/v2/v3) - they tag a different model entirely, not a variant of the 1x2
one - so a single --model flag can never scope both at once the way
report.py's one flag scopes prediction_calls and double_chance_calls
together (they share the model that produced both rows' probabilities).
Threading BTTS through report.py's _CALL_TABLES would silently report zero
rows under --model v3 and zero 1x2 rows under --model btts-v1.

What IS shared, imported rather than restated: report.py's query/render
layer (SegmentStats, _segment_stats, _bucket_rows, _render_table, the _fmt_*
helpers, Window, the confidence-score machinery) makes no assumption about
which table or which model-version namespace it is pointed at - it already
takes `table` as a plain string argument. Only the things that ARE
1x2-shaped stay in report.py: _CALL_TABLES, the two-arm headline, the
double-chance overlap/main-pick sections. This file supplies its own single
CallTable (baseline 1/2, no double-chance sibling - BTTS is already binary)
and calls report.print_summary() directly.

    python backtest/btts_report.py                  # all time
    python backtest/btts_report.py --today
    python backtest/btts_report.py --week
    python backtest/btts_report.py --model btts-backfill
    python backtest/btts_report.py --rule max_prob   # just the one rule
"""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta

import psycopg

import selection
from report import (
    DEFAULT_BTTS_MODEL_VERSION,
    POSTGRES_URL,
    CallTable,
    Window,
    print_summary,
)

#: btts_calls' own CallTable: baseline 1/2 (a model that knows nothing still
#: says "yes" half the time), no dc_prob_sql - a double-chance framing of an
#: already-binary market is not a second view, it is the same bet relabelled.
BTTS_TABLE = CallTable("both teams to score", "btts_calls", 1 / 2)

_WINDOWS = {
    "today": lambda: Window.last("last 24 hours", timedelta(days=1)),
    "week": lambda: Window.last("last 7 days", timedelta(days=7)),
}


def print_btts_report(
    window: Window,
    model_version: str = DEFAULT_BTTS_MODEL_VERSION,
    call_rule: str = selection.DEFAULT_RULE,
) -> None:
    scoped = window.for_model(model_version, call_rule)
    print(f"BTTS model performance - {scoped.label} (kickoffs)")
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM btts_calls WHERE model_version = %s AND call_rule = %s",
            (model_version, call_rule),
        )
        if cur.fetchone()[0] == 0:
            print(
                f"no graded {model_version} BTTS predictions under rule "
                f"{call_rule} yet - run backtest/grade_calls.py --market "
                f"btts --model {model_version} --rule {call_rule} first"
            )
            _print_other_versions(cur, model_version)
            return
        print_summary(cur, BTTS_TABLE, scoped)


def print_full_btts_report(
    window: Window,
    model_version: str = DEFAULT_BTTS_MODEL_VERSION,
    rules: tuple[str, ...] = selection.RULES,
) -> None:
    """Every rule's section in turn - report.print_full_report's shape,
    minus the rule-overlap section, which only makes sense for a DC sibling
    this market doesn't have."""
    for i, call_rule in enumerate(rules):
        if i:
            print("\n" + "=" * 72)
        print_btts_report(window, model_version, call_rule)


def _print_other_versions(cur, model_version: str) -> None:
    """report.py's twin: a graded-empty report reads identically to a broken
    pipeline unless it names what IS graded."""
    cur.execute(
        "SELECT model_version || '/' || call_rule, count(*) "
        "FROM btts_calls WHERE model_version <> %s GROUP BY 1 ORDER BY 1",
        (model_version,),
    )
    rows = cur.fetchall()
    if rows:
        print("\ngraded BTTS model/rule combinations that do exist:")
        for tag, n in rows:
            print(f"  {tag}: {n}")
    else:
        print(
            "\nno BTTS predictions graded at all yet - run "
            "backtest/backfill_predictions.py --model btts and "
            "backtest/grade_calls.py --market btts first"
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graded BTTS performance report for one model/rule.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--today", dest="window", action="store_const", const="today",
        help="stats for fixtures that kicked off in the last 24 hours",
    )
    group.add_argument(
        "--week", dest="window", action="store_const", const="week",
        help="stats for fixtures that kicked off in the last 7 days",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_BTTS_MODEL_VERSION,
        help=f"btts_calls.model_version to report on (default: "
        f"{DEFAULT_BTTS_MODEL_VERSION})",
    )
    parser.add_argument(
        "--rule",
        default=None,
        choices=selection.RULES,
        help="selection rule to report on. Default: every rule, in one "
        "report with a section per rule.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args(sys.argv[1:])
    window = _WINDOWS[args.window]() if args.window else Window.all_time()
    if args.rule:
        print_btts_report(window, args.model, args.rule)
    else:
        print_full_btts_report(window, args.model)


if __name__ == "__main__":
    main()
