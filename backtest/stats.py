"""Displays model prediction stats for a time window.

    python backtest/stats.py --today    # kickoffs in the last 24 hours
    python backtest/stats.py --week     # kickoffs in the last 7 days
    python backtest/stats.py            # all time
    python backtest/stats.py --model v1 # the pre-switch v1 record

Thin CLI over backtest/report.py, which holds the actual summary logic (hit
rate vs. model confidence, model-vs-market Brier, paper P&L, edge-size
buckets, each broken out by flagged/unflagged segment). Read-only; run
backtest/grade_calls.py first to grade any newly finished fixtures.

Reports one model version at a time, defaulting to whatever inference-py
serves (SERVED_MODEL_VERSION, currently v2). The v1 rows graded before the
serving switch stay queryable via --model v1, but are never blended into the
v2 numbers.

Windows are relative to kickoff, not to when a row was graded - see
report.Window. --today is therefore "matches played in the last 24h", which
on a quiet midweek day can legitimately be zero rows.
"""
from __future__ import annotations

import argparse
from datetime import timedelta

from report import DEFAULT_MODEL_VERSION, Window, print_report

_WINDOWS = {
    "today": lambda: Window.last("last 24 hours", timedelta(days=1)),
    "week": lambda: Window.last("last 7 days", timedelta(days=7)),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display model prediction stats for a time window.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--today",
        dest="window",
        action="store_const",
        const="today",
        help="stats for fixtures that kicked off in the last 24 hours",
    )
    group.add_argument(
        "--week",
        dest="window",
        action="store_const",
        const="week",
        help="stats for fixtures that kicked off in the last 7 days",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_VERSION,
        help=f"model version to report on (default: {DEFAULT_MODEL_VERSION})",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    window = _WINDOWS[args.window]() if args.window else Window.all_time()
    print_report(window, args.model)


if __name__ == "__main__":
    main()
