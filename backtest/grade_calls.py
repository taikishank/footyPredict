"""Grades finished fixtures against the model's pre-kickoff predictions.

For each fixture whose result is now known and that doesn't yet have a
prediction_calls row, picks the `edges` snapshot closest to kickoff as "the
call", compares it to what actually happened, and records a graded row:
whether the called side won, the Brier score of the full model (and market)
distribution, and a flat-stake paper P&L against the price recorded at that
snapshot.

Grades one model version only - the one inference-py is serving
(config.SERVED_MODEL_VERSION, currently v3), or whatever --model names.
`edges` still holds the v1 and v2 snapshots from before their switches; they
are never mixed into a v3 record, and the version that produced a graded call
is recorded on the row.

Every fixture the model priced is graded, not only the ones it flagged an
edge on - grading a self-selected slice can't tell you whether the model is
calibrated in general. The flagged ones stay distinguishable: each row
records `flagged` (was the graded snapshot itself flagged) and
`ever_flagged` (was ANY pre-kickoff snapshot flagged, since edges is an
append-only poll-by-poll log and a fixture can flip in and out of flagged as
odds move). backtest/report.py breaks the summary out along that split - see
PROJECT_SPEC.md Phase 6.

Run manually or on a schedule (cron); safe to re-run since already-graded
fixtures are excluded by the ungraded-fixtures query and the insert's
ON CONFLICT guard.

Alongside prediction_calls, also writes double_chance_calls: the same
snapshot graded as a double-chance bet. The two are INDEPENDENT arms, each
selecting its own side. double_chance_calls picks dc_label over all three
labels (1X excludes away_win, X2 excludes home_win, 12 excludes draw) and is
NOT derived from call_side. While it was ("the pair of outcomes excluding
call_side"), dc_correct was the exact negation of prediction_calls.correct,
so the DC hit rate was always 1 - the single hit rate and "v2 is better at
double chance" only ever restated "v2 is worse at singles". Verified on the
v2/v3 backfill: the two arms agreed on 0 of 43 fixtures, for both models.

WHICH side either arm takes is now a named selection rule rather than an
expression in this file - see backtest/selection.py - and the rule is
recorded on the graded row as `call_rule`. model_version and call_rule are
separate axes: the first says which artifact produced the probabilities, the
second says what was done with them. A fixture is graded once per
(model_version, call_rule), and because latest_edge() does not consult the
rule, both arms grade the SAME snapshot - so a difference between them is
selection and nothing else. Running with no --rule grades every rule, so a
newly added arm backfills itself.

excludes_outcome and dc_label still describe the bet correctly, so the table
needs no schema change - excludes_outcome is simply no longer equal to
prediction_calls.call_side.

No rule here ever refuses a bet, and for max_edge that is structural rather
than a property of this data: the model's three probabilities sum to 1 and
the de-vigged market's do too, so the three edges sum to exactly 0 and at
least one is always non-negative (same for DC, whose two-of-three
probabilities sum to 2 on both sides, so its three label edges also sum to
0). Measured: 777 of 777 v3 snapshots. A max-positive-edge selection
therefore always names a side, and "the model found an edge here" is not
information. Gated rules that CAN refuse are the obvious next arms; every
rule shipped so far stakes the whole book, which keeps the arms comparable
on one sample rather than on differently-sized slices.
ROI over all graded rows is consequently not a strategy's ROI; only rows
clearing EDGE_FLAG_THRESHOLD are stakes a strategy would place, which is why
backtest/report.py gates its ROI figure on that and grades everything else
for calibration only.

The DC price is SYNTHESIZED, not observed. We hold no double-chance odds, so
the bet is priced by applying the fixture's own 1X2 overround to the
de-vigged DC probability:

    overround = 1/home_price + 1/draw_price + 1/away_price   (~1.05 typical)
    dc_price  = (1.0 / dc_market_prob) / overround

This assumes books apply the same proportional margin to double chance as to
1X2. Real DC margins are typically slightly LOWER than that, so this biases
marginally in the DC arm's favour. It replaces a much larger bias in the
same direction: pricing DC at a flat 1/dc_market_prob gave the DC arm a
zero-margin price while singles settled against real prices carrying the
book's overround - measured at ~5 points of ROI of pure artifact, larger
than any plausible model edge, which made the two arms incomparable on ROI.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import timedelta

import psycopg
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

POSTGRES_URL = os.environ["POSTGRES_URL"]

# Default to whatever inference-py is serving, so grading follows the switch
# automatically rather than needing to be remembered separately. Imported from
# report.py rather than restated here: this file and that one used to declare
# the same os.environ.get with the same hand-copied fallback, and a promotion
# moves a literal that appears twice exactly once.
from report import DEFAULT_MODEL_VERSION  # noqa: E402

import selection  # noqa: E402

# Buffer after kickoff before treating a fixture as done - gives ingestor-go's
# batch pipeline time to backfill the final score before we go looking for it.
_GRADE_AFTER = timedelta(hours=2.5)

_OUTCOMES = ("home_win", "draw", "away_win")
_EDGE_COL = {"home_win": "edge_home", "draw": "edge_draw", "away_win": "edge_away"}
_MODEL_COL = {"home_win": "model_home", "draw": "model_draw", "away_win": "model_away"}
_MARKET_COL = {"home_win": "market_home", "draw": "market_draw", "away_win": "market_away"}
_PRICE_COL = {"home_win": "home_price", "draw": "draw_price", "away_win": "away_price"}

# Double-chance label for the bet that excludes a given singleton outcome
# (e.g. excluding away_win leaves home-or-draw, i.e. "1X").
_DC_LABEL = {"away_win": "1X", "home_win": "X2", "draw": "12"}


def find_ungraded_finished_fixtures(
    conn: psycopg.Connection, model_version: str, call_rule: str
) -> list[int]:
    """Finished fixtures with no graded row *for this model version under
    this selection rule*. A fixture already graded under v1 is still ungraded
    under v2, and one already graded under max_edge is still ungraded under
    max_prob - the two arms are separate records of the same snapshot."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.fixture_id
            FROM fixtures f
            WHERE f.result IS NOT NULL
              AND f.starting_at < now() - %s
              -- Only fixtures this model actually priced. Without it the
              -- scan returns every finished fixture ever ingested (~40k)
              -- and reports them all as "skipped", which drowns the real
              -- signal - a fixture the model DID price but only after
              -- kickoff. It also stopped being free once grading looped
              -- over rules: the same 40k were re-walked once per arm.
              AND EXISTS (
                  SELECT 1 FROM edges e
                  WHERE e.fixture_id = f.fixture_id
                    AND e.model_version = %s
              )
              AND NOT EXISTS (
                  SELECT 1 FROM prediction_calls pc
                  WHERE pc.fixture_id = f.fixture_id
                    AND pc.model_version = %s
                    AND pc.call_rule = %s
              )
            """,
            (_GRADE_AFTER, model_version, model_version, call_rule),
        )
        return [row[0] for row in cur.fetchall()]


def latest_edge(
    conn: psycopg.Connection, fixture_id: int, starting_at, model_version: str
) -> dict | None:
    """This model version's snapshot closest to kickoff - the last thing it
    believed before the match was no longer "upcoming" - flagged or not."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT computed_at, model_home, model_draw, model_away,
                   market_home, market_draw, market_away,
                   edge_home, edge_draw, edge_away,
                   home_price, draw_price, away_price, flagged
            FROM edges
            WHERE fixture_id = %s AND computed_at < %s AND model_version = %s
            ORDER BY computed_at DESC
            LIMIT 1
            """,
            (fixture_id, starting_at, model_version),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = (
            "computed_at", "model_home", "model_draw", "model_away",
            "market_home", "market_draw", "market_away",
            "edge_home", "edge_draw", "edge_away",
            "home_price", "draw_price", "away_price", "flagged",
        )
        return dict(zip(keys, row))


def ever_flagged(
    conn: psycopg.Connection, fixture_id: int, starting_at, model_version: str
) -> bool:
    """Whether the model flagged an edge at any point before kickoff, even if
    the snapshot it settled on by kickoff no longer was."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1 FROM edges
                WHERE fixture_id = %s AND computed_at < %s AND flagged = true
                  AND model_version = %s
            )
            """,
            (fixture_id, starting_at, model_version),
        )
        return bool(cur.fetchone()[0])


def _dc_price(dc_market_prob: float, call: dict) -> float | None:
    """A synthesized double-chance price: the de-vigged DC probability
    re-loaded with the fixture's own 1X2 margin.

    We hold no double-chance odds. `market_*` is de-vigged in
    ingestor-go/internal/odds/parse.go (the three implied probabilities sum
    to exactly 1.0000), so 1/dc_market_prob is a FAIR price with zero
    margin, while the singles arm settles against the real home/draw/away
    price, which carries the book's overround (measured: 1/price sums to
    ~1.0514). Pricing the two arms on those terms handed DC ~5 points of
    ROI as an artifact of the de-vigging, which is larger than any model
    edge either arm is trying to detect.

    The margin is taken per fixture, not as a global constant, because the
    overround varies by fixture and league and a constant would just move
    the artifact rather than remove it. Assumes books apply the same
    proportional margin to DC as to 1X2; real DC margins are usually a
    little lower, so this still tilts marginally toward the DC arm - but by
    the residual difference between two margins rather than by a whole
    margin.
    """
    if dc_market_prob <= 0:
        return None
    fair = 1.0 / dc_market_prob
    prices = [call[_PRICE_COL[o]] for o in _OUTCOMES]
    if any(p is None or float(p) <= 0 for p in prices):
        # No 1X2 prices to read a margin from. Fall back to the fair price,
        # which flatters this row's DC P&L by roughly the missing margin -
        # the same bias the per-fixture overround exists to remove. Rare
        # enough to leave in rather than drop the row.
        return fair
    overround = sum(1.0 / float(p) for p in prices)
    return fair / overround


def _brier_score(probs_by_outcome: dict[str, float], actual_result: str) -> float:
    return sum(
        (probs_by_outcome[o] - (1.0 if o == actual_result else 0.0)) ** 2
        for o in _OUTCOMES
    )


def grade_fixture(
    conn: psycopg.Connection, fixture_id: int, model_version: str, call_rule: str
) -> bool:
    """Grades one fixture's call under one selection rule, if the model ever
    priced it before kickoff. Returns True if a prediction_calls row was
    written.

    Every rule grades the SAME snapshot - latest_edge() does not consult the
    rule - so two arms over one fixture differ only in which side they took.
    That is the whole point of keying on the rule instead of reconstructing
    the probabilities once per arm."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT starting_at, result FROM fixtures WHERE fixture_id = %s",
            (fixture_id,),
        )
        starting_at, actual_result = cur.fetchone()

    call = latest_edge(conn, fixture_id, starting_at, model_version)
    if call is None:
        return False  # this model never priced this fixture before kickoff

    was_flagged = bool(call["flagged"])
    was_ever_flagged = was_flagged or ever_flagged(
        conn, fixture_id, starting_at, model_version
    )

    model_probs = {o: float(call[_MODEL_COL[o]]) for o in _OUTCOMES}
    market_probs = {o: float(call[_MARKET_COL[o]]) for o in _OUTCOMES}

    # Which side this rule backs. The rule itself, and why "the model found
    # an edge" is not information, live in backtest/selection.py.
    call_side = selection.SINGLE[call_rule](model_probs, market_probs)
    model_prob = float(call[_MODEL_COL[call_side]])
    market_prob = float(call[_MARKET_COL[call_side]])
    edge = float(call[_EDGE_COL[call_side]])
    price = call[_PRICE_COL[call_side]]
    price = float(price) if price is not None else None

    correct = call_side == actual_result

    brier_score = _brier_score(model_probs, actual_result)
    market_brier_score = _brier_score(market_probs, actual_result)

    staked_units = 1.0
    pnl_units = None
    if price is not None:
        pnl_units = ((price - 1.0) if correct else -1.0) * staked_units

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prediction_calls (
                fixture_id, edge_computed_at, call_side, model_prob, market_prob,
                edge, price, actual_result, correct, brier_score, market_brier_score,
                staked_units, pnl_units, flagged, ever_flagged, model_version,
                call_rule
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fixture_id, model_version, call_rule) DO NOTHING
            """,
            (
                fixture_id, call["computed_at"], call_side, model_prob, market_prob,
                edge, price, actual_result, correct, brier_score, market_brier_score,
                staked_units, pnl_units, was_flagged, was_ever_flagged,
                model_version, call_rule,
            ),
        )

    # The DC arm selects its own side under the same named rule, never
    # derived from call_side - deriving it made dc_correct the exact negation
    # of `correct`, so the arm measured nothing the singles arm had not.
    dc_excluded = selection.DOUBLE_CHANCE[call_rule](model_probs, market_probs)
    dc_label = _DC_LABEL[dc_excluded]
    dc_model_prob = 1.0 - float(call[_MODEL_COL[dc_excluded]])
    dc_market_prob = 1.0 - float(call[_MARKET_COL[dc_excluded]])
    dc_edge = dc_model_prob - dc_market_prob
    dc_correct = actual_result != dc_excluded  # wins if either covered outcome hits
    dc_price = _dc_price(dc_market_prob, call)
    dc_pnl_units = None
    if dc_price is not None:
        dc_pnl_units = ((dc_price - 1.0) if dc_correct else -1.0) * staked_units
    dc_brier_score = (dc_model_prob - (1.0 if dc_correct else 0.0)) ** 2
    dc_market_brier_score = (dc_market_prob - (1.0 if dc_correct else 0.0)) ** 2

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO double_chance_calls (
                fixture_id, edge_computed_at, dc_label, excludes_outcome, model_prob,
                market_prob, edge, implied_price, actual_result, correct, brier_score,
                market_brier_score, staked_units, pnl_units, flagged, ever_flagged,
                model_version, call_rule
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (fixture_id, model_version, call_rule) DO NOTHING
            """,
            (
                fixture_id, call["computed_at"], dc_label, dc_excluded, dc_model_prob,
                dc_market_prob, dc_edge, dc_price, actual_result, dc_correct,
                dc_brier_score, dc_market_brier_score, staked_units, dc_pnl_units,
                was_flagged, was_ever_flagged, model_version, call_rule,
            ),
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grade finished fixtures against one model version's predictions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_VERSION,
        help=f"edges.model_version to grade (default: {DEFAULT_MODEL_VERSION})",
    )
    parser.add_argument(
        "--rule",
        action="append",
        choices=selection.RULES,
        help="selection rule to grade; repeatable. Default: every rule, so a "
        "new arm backfills itself on the next run rather than needing to be "
        "remembered separately.",
    )
    args = parser.parse_args()

    rules = args.rule or list(selection.RULES)

    with psycopg.connect(POSTGRES_URL) as conn:
        for call_rule in rules:
            graded = 0
            skipped = 0
            # Re-queried per rule: a fixture graded under max_edge is still
            # ungraded under max_prob, so the two arms have different backlogs.
            for fixture_id in find_ungraded_finished_fixtures(
                conn, args.model, call_rule
            ):
                if grade_fixture(conn, fixture_id, args.model, call_rule):
                    graded += 1
                else:
                    skipped += 1
            print(
                f"[{args.model}/{call_rule}] graded {graded} prediction(s); "
                f"{skipped} finished fixture(s) had no pre-kickoff "
                f"{args.model} snapshot",
                file=sys.stderr,
            )
        conn.commit()


if __name__ == "__main__":
    main()
