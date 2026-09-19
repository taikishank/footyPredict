"""Displays one row per graded match: both arms' pick, confidence, and the
P&L/ROI a flat 1-unit wager on that pick actually returned.

    python backtest/match_report.py --today    # kickoffs in the last 24 hours
    python backtest/match_report.py --week     # kickoffs in the last 7 days
    python backtest/match_report.py            # all time
    python backtest/match_report.py --model v2 # the pre-v3 record

Same Window/model-version CLI as backtest/stats.py, and it shares Window and
DEFAULT_MODEL_VERSION with report.py so "last 7 days" and "model v2" mean the
same thing in both tools. Where stats.py aggregates into segment tables, this
prints the row report.py deliberately never does: one line per fixture, both
arms side by side, so a specific match's call can be checked without a SQL
console.

Confidence is model_prob as graded - the probability the model put on the
side it actually took - not the frontend's rescaled confidence score (see
report._confidence_score_sql). That rescaling exists to make the badge
thresholds comparable across a 1/3 and a 2/3 baseline; here there is no badge,
just the number that was staked on, so the raw probability is the more direct
answer to "how confident was the model in this specific pick".

P&L and ROI are both per-row, not aggregated: pnl_units is the raw profit on
that fixture's own 1-unit bet (+1.85u), and ROI is pnl_units / staked_units
for the same bet (+185%), a loss reading -1.00u / -100%. Both are priced
whether or not the row cleared EDGE_FLAG_THRESHOLD - unlike report.py's ROI,
which is gated to the staked segment because an ungated aggregate would be
"the ROI of betting every fixture" and read like a strategy's number. A
single row has no such ambiguity: it is always "what this one bet would have
paid", which is exactly what belongs next to a specific match. Rows with no
price on record (see grade_calls.py's DC synthesis and its None-price
fallback) show P&L and ROI as the same em-dash report.py uses for "no value
here", not a zero.
"""
from __future__ import annotations

import argparse
from datetime import timedelta

import psycopg

import selection
from report import (
    DEFAULT_MODEL_VERSION,
    POSTGRES_URL,
    Window,
    _ABSENT,
    _render_table,
)

_WINDOWS = {
    "today": lambda: Window.last("last 24 hours", timedelta(days=1)),
    "week": lambda: Window.last("last 7 days", timedelta(days=7)),
}

# call_side / excludes_outcome use the DB's snake_case outcome names;
# dc_label is already the short market convention (1X/X2/12). Only the
# singles side needs translating for a compact column.
_OUTCOME_LABEL = {"home_win": "Home", "draw": "Draw", "away_win": "Away"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-match table: both arms' pick, confidence, and ROI.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--today",
        dest="window",
        action="store_const",
        const="today",
        help="matches that kicked off in the last 24 hours",
    )
    group.add_argument(
        "--week",
        dest="window",
        action="store_const",
        const="week",
        help="matches that kicked off in the last 7 days",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_VERSION,
        help=f"model version to report on (default: {DEFAULT_MODEL_VERSION}). "
        "Free text rather than a fixed choice list, because the set of graded "
        "versions lives in the database and grows with every backfill - "
        "--list-versions prints what is actually there.",
    )
    parser.add_argument(
        "--list-versions",
        action="store_true",
        help="print the graded model/rule combinations and exit",
    )
    parser.add_argument(
        "--rule",
        default=selection.DEFAULT_RULE,
        choices=selection.RULES,
        help=f"selection rule to report on (default: {selection.DEFAULT_RULE}). "
        "A table mixing rules would list each match twice with different "
        "picks, so this names one.",
    )
    return parser.parse_args(argv)


def _roi(pnl, staked) -> str:
    if pnl is None or not staked:
        return _ABSENT
    return f"{float(pnl) / float(staked):+.1%}"


def _pnl(pnl) -> str:
    if pnl is None:
        return _ABSENT
    return f"{float(pnl):+.2f}u"


def print_match_report(
    window: Window,
    model_version: str = DEFAULT_MODEL_VERSION,
    call_rule: str = selection.DEFAULT_RULE,
) -> None:
    scoped = window.for_model(model_version, call_rule)
    print(f"match report - {scoped.label} (kickoffs), all leagues")
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT f.starting_at, f.home_name, f.away_name,
                   c.call_side, c.model_prob, c.pnl_units, c.staked_units,
                   d.dc_label, d.model_prob, d.pnl_units, d.staked_units
            FROM prediction_calls c
            JOIN double_chance_calls d USING (fixture_id, model_version, call_rule)
            JOIN fixtures f USING (fixture_id)
            WHERE ({scoped.sql})
            ORDER BY f.starting_at
            """,
            scoped.params,
        )
        rows = cur.fetchall()

    if not rows:
        print(f"no graded predictions in this window ({scoped.label})")
        # An unknown or misspelled --model looks exactly like a correct one
        # with no rows yet, so name what is actually graded rather than
        # leaving the difference to be guessed.
        print()
        print_graded_versions()
        return

    table_rows = []
    single_confs, dc_confs = [], []
    single_pnls, single_stakes, dc_pnls, dc_stakes = [], [], [], []
    for (starting_at, home, away, call_side, single_conf, single_pnl, single_staked,
         dc_label, dc_conf, dc_pnl, dc_staked) in rows:
        table_rows.append([
            f"{starting_at:%Y-%m-%d %H:%M}",
            f"{home} v {away}",
            _OUTCOME_LABEL[call_side],
            f"{float(single_conf):.1%}",
            _pnl(single_pnl),
            _roi(single_pnl, single_staked),
            dc_label,
            f"{float(dc_conf):.1%}",
            _pnl(dc_pnl),
            _roi(dc_pnl, dc_staked),
        ])
        single_confs.append(float(single_conf))
        dc_confs.append(float(dc_conf))
        if single_pnl is not None:
            single_pnls.append(float(single_pnl))
            single_stakes.append(float(single_staked))
        if dc_pnl is not None:
            dc_pnls.append(float(dc_pnl))
            dc_stakes.append(float(dc_staked))

    # Totals row: ROI here is the aggregate over every row in the window
    # (sum P&L / sum staked), not an average of the per-row ROIs above - a
    # few big wins on small stakes would otherwise swamp the aggregate, the
    # same distinction report.py's segment ROI draws.
    table_rows.append([
        "TOTAL",
        f"{len(rows)} matches",
        "-",
        f"{sum(single_confs) / len(single_confs):.1%}",
        _pnl(sum(single_pnls)) if single_pnls else _ABSENT,
        _roi(sum(single_pnls), sum(single_stakes)) if single_pnls else _ABSENT,
        "-",
        f"{sum(dc_confs) / len(dc_confs):.1%}",
        _pnl(sum(dc_pnls)) if dc_pnls else _ABSENT,
        _roi(sum(dc_pnls), sum(dc_stakes)) if dc_pnls else _ABSENT,
    ])

    _render_table(
        ["kickoff", "match", "pick", "conf", "P&L", "ROI",
         "DC pick", "DC conf", "DC P&L", "DC ROI"],
        table_rows,
    )


def print_graded_versions() -> None:
    """What is actually in prediction_calls, so "which version?" has an
    answer that does not require reading grade_calls.py or guessing.

    `v3` and `v3-backfill` are both v3 - the same model over different
    fixtures - so the name alone does not say which one holds the rows you
    want. The counts and kickoff ranges do.
    """
    with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.model_version, c.call_rule, count(*),
                   min(f.starting_at)::date, max(f.starting_at)::date
            FROM prediction_calls c JOIN fixtures f USING (fixture_id)
            GROUP BY 1, 2 ORDER BY 1, 2
            """
        )
        rows = cur.fetchall()
    if not rows:
        print("nothing graded yet - run backtest/grade_calls.py first")
        return
    _render_table(
        ["--model", "--rule", "matches", "first kickoff", "last kickoff"],
        [[v, r, str(n), str(lo), str(hi)] for v, r, n, lo, hi in rows],
    )


def main() -> None:
    args = parse_args()
    if args.list_versions:
        print_graded_versions()
        return
    window = _WINDOWS[args.window]() if args.window else Window.all_time()
    print_match_report(window, args.model, args.rule)


if __name__ == "__main__":
    main()
