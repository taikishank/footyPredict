"""Displays model prediction stats for a time window.

    python backtest/stats.py --today    # kickoffs in the last 24 hours
    python backtest/stats.py --week     # kickoffs in the last 7 days
    python backtest/stats.py            # all time
    python backtest/stats.py --model v2 # the pre-v3 record
    python backtest/stats.py --compare v2 v3   # the two head to head

Thin CLI over backtest/report.py, which holds the actual summary logic (hit
rate vs. model confidence, model-vs-market Brier, paper P&L, ROI, edge-size
buckets, each broken out by flagged/unflagged segment). It prints a headline
table of the two arms - whole record, staked subset and the below-gate
remainder the edge gate refused - then one table per
grouping per arm, and finally an overlap table pricing the two arms as the
single 2-unit position they actually are when both are run together. Read-only; run backtest/grade_calls.py first to grade any
newly finished fixtures.

Reports one model version at a time, defaulting to whatever inference-py
serves (SERVED_MODEL_VERSION, currently v3). Rows graded under an earlier
serving model stay queryable via --model v1 / --model v2, but are never
blended into the v3 numbers - there are now three tiers in prediction_calls,
and each switch adds one rather than replacing what came before.

--compare puts two versions head to head, and does it only on the fixtures
*both* of them graded. That restriction is the whole value: each version's
record covers the period it was serving, so reading two single-version reports
side by side compares September with October and calls the difference a model.
The paired view removes the calendar, the league mix and the fixture
difficulty at once, and reports a bootstrap interval on the Brier difference
using ml/rating_ablate.py's convention, so a graded margin and an offline one
are read on the same scale.

The cost of that restriction is that the intersection is usually empty: two
versions only overlap where both priced the same matches, which happens after
a reconstruction (the `v2-backfill` tier) or during shadow scoring, not simply
because both have rows. --compare says which versions have graded rows and
what to do about it rather than printing an empty table. Before an overlap
exists, the model-vs-model question is answered offline by
inference-py/scripts/compare_v2_v3_served.py.

Windows are relative to kickoff, not to when a row was graded - see
report.Window. --today is therefore "matches played in the last 24h", which
on a quiet midweek day can legitimately be zero rows.
"""
from __future__ import annotations

import argparse
from datetime import timedelta

from report import DEFAULT_MODEL_VERSION, Window, print_comparison, print_report

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
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "ARM"),
        help="two model versions to put head to head on the fixtures both "
             "graded, e.g. --compare v2 v3. Ignores --model.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    window = _WINDOWS[args.window]() if args.window else Window.all_time()
    if args.compare:
        baseline, arm = args.compare
        print_comparison(window, baseline, arm)
    else:
        print_report(window, args.model)


if __name__ == "__main__":
    main()
