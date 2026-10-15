"""The same graded-performance report as backtest/report.py, scoped to one or
more leagues.

report.py prints "all leagues" by construction - its headline, per-arm,
overlap and main-pick sections never filter on fixtures.league_id. This file
adds exactly that one filter and reuses every query and table-rendering
function report.py already has: _segment_stats, _bucket_rows, print_headline,
print_summary, print_overlap, print_main_pick and print_rule_overlap all take
a Window, and Window is already the mechanism report.py uses to scope a
report to a time slice (see Window.last / Window.for_model). Scoping to a
league is the same move: AND an extra predicate onto Window.sql before
for_model() ANDs in model_version and call_rule, so nothing downstream needs
to know leagues exist.

League ids are SportMonks' own, taken from sm_league_id_to_raw_name_mapping.txt
at the repo root - the same ids fixtures.league_id stores (see
inference-py/app/db.py, which reads league_id straight off the SportMonks
fixture payload).

    python backtest/league_report.py --pl
    python backtest/league_report.py --pl --sa
    python backtest/league_report.py --all
    python backtest/league_report.py --all --model v2 --rule max_prob
"""
from __future__ import annotations

import argparse

import psycopg

import report
import selection
from report import POSTGRES_URL, Window

#: SportMonks league id -> (flag dest, display name), from
#: sm_league_id_to_raw_name_mapping.txt.
_LEAGUES: list[tuple[int, str, str]] = [
    (8, "pl", "Premier League"),
    (82, "bun", "Bundesliga"),
    (301, "l1", "Ligue 1"),
    (384, "sa", "Serie A"),
    (564, "ll", "La Liga"),
]


def _with_league(window: Window, league_id: int, league_label: str) -> Window:
    """ANDs a fixtures.league_id predicate onto an existing Window.

    league_id is always one of the fixed literals in _LEAGUES, never user
    input, so interpolating it as a param (not an f-string) is just the
    normal Window convention, not a safety requirement here.
    """
    return Window(
        f"{window.label}, league {league_label}",
        f"({window.sql}) AND f.league_id = %s",
        window.params + (league_id,),
    )


def print_league_report(
    window: Window,
    league_label: str,
    model_version: str,
    call_rule: str,
) -> None:
    """report.print_report(), with the header's "all leagues" replaced by
    the one league this call is actually scoped to. Everything below the
    header is report.py's own functions, unchanged - the formatted tables
    are identical to report.py's, just over a league-filtered row set.
    """
    scoped = window.for_model(model_version, call_rule)
    print(f"model performance - {scoped.label} (kickoffs), {league_label}")
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
            report._print_other_versions(cur, model_version)
            return
        if not report.print_headline(cur, scoped):
            print(f"no graded predictions in this window ({scoped.label})")
            return
        for call_table in report._CALL_TABLES:
            report.print_summary(cur, call_table, scoped)
        report.print_overlap(cur, scoped)
        report.print_main_pick(cur, scoped)


def print_full_league_report(
    window: Window,
    league_label: str,
    model_version: str,
    rules: tuple[str, ...] = selection.RULES,
) -> None:
    """report.print_full_report(), scoped to one league: every rule's
    section for that league, then that league's rule-overlap section."""
    for i, call_rule in enumerate(rules):
        if i:
            print("\n" + "=" * 72)
        print_league_report(window, league_label, model_version, call_rule)
    if len(rules) > 1:
        print("\n" + "=" * 72)
        with psycopg.connect(POSTGRES_URL) as conn, conn.cursor() as cur:
            report.print_rule_overlap(cur, model_version, window)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Graded-performance report (same tables as "
        "backtest/report.py), scoped to one or more leagues.",
    )
    parser.add_argument(
        "--all", action="store_true", help="report every league (default if "
        "no league flag is given)"
    )
    for league_id, dest, name in _LEAGUES:
        parser.add_argument(
            f"--{dest}", action="store_true", help=f"report {name} (league_id={league_id})"
        )
    parser.add_argument(
        "--model",
        default=report.DEFAULT_MODEL_VERSION,
        help=f"model version to report on (default: {report.DEFAULT_MODEL_VERSION})",
    )
    parser.add_argument(
        "--rule",
        default=None,
        choices=selection.RULES,
        help="selection rule to report on (default: every rule, like "
        "report.py with no --rule)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    selected = [dest for _id, dest, _name in _LEAGUES if getattr(args, dest)]
    if args.all or not selected:
        selected = [dest for _id, dest, _name in _LEAGUES]

    window = Window.all_time()
    by_dest = {dest: (league_id, name) for league_id, dest, name in _LEAGUES}
    for i, dest in enumerate(selected):
        league_id, name = by_dest[dest]
        if i:
            print("\n" + "=" * 72)
        league_window = _with_league(window, league_id, name)
        if args.rule:
            print_league_report(league_window, name, args.model, args.rule)
        else:
            print_full_league_report(league_window, name, args.model)


if __name__ == "__main__":
    main()
