import { ChangeDetectionStrategy, Component, OnInit, inject, output } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';

import { UpcomingFixturesService } from './upcoming-fixtures.service';
import type { UpcomingFixture } from './upcoming-fixture.model';
import type { Probabilities } from '../live-prediction/live-prediction.model';

type Outcome = 'home_win' | 'draw' | 'away_win';

const OUTCOMES: Outcome[] = ['home_win', 'draw', 'away_win'];

const OUTCOME_LABEL: Record<Outcome, string> = {
  home_win: 'Home win',
  draw: 'Draw',
  away_win: 'Away win',
};

// Double-chance label for the bet that excludes a given singleton outcome
// (e.g. excluding away_win leaves home-or-draw, i.e. "1X"). Mirrors
// backtest/grade_calls.py's _DC_LABEL. Keyed on the EXCLUDED outcome, which
// the double-chance pick now chooses for itself - it is no longer whichever
// side the singleton pick called.
const DC_LABEL: Record<Outcome, string> = {
  away_win: '1X',
  home_win: 'X2',
  draw: '12',
};

// Plain-language gloss for the same excluded-outcome keying as DC_LABEL, so
// the UI can spell out which two outcomes the double-chance % actually
// covers instead of relying on the 1X/X2/12 shorthand alone.
const DC_DESCRIPTION: Record<Outcome, string> = {
  away_win: 'home win or draw',
  home_win: 'draw or away win',
  draw: 'home or away win (no draw)',
};

// The two selection rules, mirroring backtest/selection.py. They are the
// same two rules the backtest grades as `call_rule`, so a pick shown here
// under `max_edge` means what a prediction_calls row with
// call_rule = 'max_edge' means once the fixture is graded.
//
// They exist as two columns rather than one because they disagree far more
// often than they agree: measured over 777 v3 snapshots, argmax(edge) and
// argmax(prob) named the same outcome on 147 of them (18.9%).
export type CallRule = 'max_edge' | 'max_prob';

// What each rule is actually doing, in one line, for the column subtitle.
const CALL_RULE_DESCRIPTION: Record<CallRule, string> = {
  max_edge: 'the outcome the market most underprices',
  max_prob: 'the outcome the model thinks most likely',
};

export type ConfidenceLevel = 'low' | 'medium' | 'high';

export interface Confidence {
  level: ConfidenceLevel;
  // 0-1: how far modelProb sits above the no-information baseline for this
  // bet type, rescaled so 0 = baseline (a coin-flip-equivalent call) and
  // 1 = certainty. Distinct from modelProb itself, which is an absolute
  // probability, not a measure of how much the model is telling you.
  //
  // This is what the low/medium/high tiers below bucket, but it is NOT the
  // % the badge also prints: that is modelProb, matching the `conf` column
  // of backtest/match_report.py (the probability the model put on the side
  // it actually took, which is what a stake is sized against). The two can
  // disagree - e.g. a double chance at 78% modelProb is only "medium" tier,
  // since its baseline is 2/3, not 0 - so the template labels them
  // separately ("medium confidence · 78.0% probability") rather than
  // printing the % inside the badge as if it were the same measure.
  score: number;
}

export interface ModelPick {
  // Which rule produced this pick. Carried on the object rather than only
  // implied by which column rendered it, so the badge and the graded row
  // can never disagree about what is being shown.
  rule: CallRule;
  label: string;
  modelProb: number;
  // null when the odds poller hasn't matched a bookmaker event to the
  // fixture - the pick is still shown, just without a market to beat.
  // Kept as one nullable object rather than two nullable fields so the
  // template narrows both at once.
  vsMarket: { marketProb: number; edge: number } | null;
  // Whether the edge clears EDGE_FLAG_THRESHOLD, i.e. this is one of the
  // calls the backtest module grades. Drives the highlight, not whether
  // the pick is shown at all.
  flagged: boolean;
  confidence: Confidence;
  // Plain-language gloss of which outcomes the % covers, e.g. "home win or
  // draw". Only set for the double-chance pick - the singleton pick's label
  // is already self-explanatory.
  description: string | null;
}

// Score thresholds for bucketing a continuous confidence score into a label.
const CONFIDENCE_MEDIUM_MIN = 0.3;
const CONFIDENCE_HIGH_MIN = 0.65;

// Mirrors inference-py/app/config.py's EDGE_FLAG_THRESHOLD default.
//
// This existed because fx.flagged was once the *incumbent's* flag while the
// UI showed a challenger's pick, so the two could disagree and the flag had
// to be re-derived here. Since the v2 promotion the server computes edge and
// flagged from whatever SERVED_MODEL_VERSION names (main.py's
// _market_and_edge takes the served probabilities), and as of the v3
// promotion that is v3, so both sides now read the same model.
//
// They still do not compute the same thing, and the client's answer is the
// one the UI wants. fx.flagged is max(|edge|) >= threshold over all three
// outcomes (main.py), a fixture-level "the model disagrees with the market
// somewhere" - which is true just as easily of an outcome the market
// OVERprices, i.e. one nobody would back. Each pick here gates on its own
// signed edge instead: the singleton pick on the edge of the side it calls,
// the double chance on the double-chance edge. That is the same distinction
// backtest/report.py draws between `flagged` and its stake segment.
const EDGE_FLAG_THRESHOLD = 0.05;

@Component({
  selector: 'app-upcoming-fixtures',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DatePipe, DecimalPipe],
  templateUrl: './upcoming-fixtures.component.html',
})
export class UpcomingFixturesComponent implements OnInit {
  protected readonly fixturesService = inject(UpcomingFixturesService);

  // Emits a fixture id so the parent can switch to the Live tab pre-filled
  // with it (see PROJECT_SPEC.md Phase 4: "hand off to the Live tab").
  readonly watchLive = output<number>();

  ngOnInit(): void {
    void this.fixturesService.load();
  }

  protected watch(fixtureId: number): void {
    this.watchLive.emit(fixtureId);
  }

  protected confidenceBadgeClass(level: ConfidenceLevel): string {
    switch (level) {
      case 'high':
        return 'bg-emerald-950/60 text-emerald-300';
      case 'medium':
        return 'bg-slate-800 text-slate-300';
      case 'low':
        return 'bg-slate-800/60 text-slate-500';
    }
  }

  // The side each rule is calling, for every fixture the model could predict
  // at all - not just the flagged ones. Mirrors backtest/selection.py, so a
  // pick shown here means the same thing it will mean once the call gets
  // graded against the actual result under the matching `call_rule`.
  //
  // max_edge takes the largest SIGNED edge, the outcome the model thinks the
  // market most underprices. max_prob takes the model's own favourite and
  // never looks at the market. Both are shown because they are usually
  // different outcomes - 81% of the time on the v3 snapshots - and which of
  // them deserves to be called "the model's pick" is exactly what the
  // backtest is now measuring rather than assuming.
  // Reads fx.probabilities_v2, which despite the name is not v2: since the
  // v2 promotion the server mirrors the *served* model into both it and
  // fx.probabilities, and the served model is now v3 (see app/schemas.py,
  // where the field is documented as a legacy alias). The two are equal, so
  // which one this reads no longer matters - switching to fx.probabilities
  // and dropping the alias is the cleanup, and it has to happen here first
  // before the server can stop sending it.
  protected pickMaxEdge(fx: UpcomingFixture): ModelPick | null {
    return this.singlePick(fx, 'max_edge');
  }

  protected pickMaxProb(fx: UpcomingFixture): ModelPick | null {
    return this.singlePick(fx, 'max_prob');
  }

  protected ruleDescription(rule: CallRule): string {
    return CALL_RULE_DESCRIPTION[rule];
  }

  private singlePick(fx: UpcomingFixture, rule: CallRule): ModelPick | null {
    const probs = fx.probabilities_v2;
    if (!probs) return null;
    const side = this.callSide(probs, fx.market, rule);
    return this.pickFrom(
      rule,
      OUTCOME_LABEL[side],
      probs[side],
      fx.market === null ? null : fx.market[side],
      // A singleton call covers 1 of 3 outcomes, so a "no information" model
      // would still assign it 1/3.
      1 / 3,
      null,
    );
  }

  // The double-chance bet, chosen INDEPENDENTLY of the singles pick within
  // the same rule. Mirrors backtest/selection.py's DOUBLE_CHANCE selectors.
  //
  // This used to be the singles pick's complement - the pair of outcomes
  // excluding whichever side it called - which meant the card always
  // displayed a self-contradicting pair: "Pick: Home win" above "Double
  // chance: X2 (draw or away win)". It was also not a second opinion in any
  // useful sense, since the double chance won exactly when the pick lost.
  // The backtest hit the same problem from the other end: the DC hit rate
  // was always 1 - the singleton hit rate, so the arm measured nothing new.
  // See grade_calls.py's module docstring.
  //
  // max_edge now excludes the outcome with the most NEGATIVE edge - the one
  // the market most overprices; max_prob excludes the one it thinks least
  // likely. Under both, argmax and argmin of the same three values are never
  // the same outcome, so the double chance always COVERS its column's
  // singles pick rather than contradicting it.
  protected dcPickMaxEdge(fx: UpcomingFixture): ModelPick | null {
    return this.doubleChancePick(fx, 'max_edge');
  }

  protected dcPickMaxProb(fx: UpcomingFixture): ModelPick | null {
    return this.doubleChancePick(fx, 'max_prob');
  }

  private doubleChancePick(fx: UpcomingFixture, rule: CallRule): ModelPick | null {
    const probs = fx.probabilities_v2;
    if (!probs) return null;
    const excluded = this.doubleChanceExcludedSide(probs, fx.market, rule);
    return this.pickFrom(
      rule,
      DC_LABEL[excluded],
      1 - probs[excluded],
      fx.market === null ? null : 1 - fx.market[excluded],
      // A double-chance bet covers 2 of the 3 outcomes, so a "no
      // information" model would still assign it 2/3.
      2 / 3,
      DC_DESCRIPTION[excluded],
    );
  }

  /**
   * Assembles one pick from a side that has already been chosen. Both arms
   * share this, so both gate on their OWN edge (model minus market for the
   * bet actually being shown) against EDGE_FLAG_THRESHOLD - for the double
   * chance that is the double-chance edge, not the singleton edge of some
   * other outcome. Only a positive edge is worth flagging: a large negative
   * edge just means the market likes that bet more than the model does.
   */
  private pickFrom(
    rule: CallRule,
    label: string,
    modelProb: number,
    marketProb: number | null,
    baseline: number,
    description: string | null,
  ): ModelPick {
    const confidence = this.confidenceFor(modelProb, baseline);
    if (marketProb === null) {
      return {
        rule,
        label,
        modelProb,
        vsMarket: null,
        flagged: false,
        confidence,
        description,
      };
    }
    const edge = modelProb - marketProb;
    return {
      rule,
      label,
      modelProb,
      vsMarket: { marketProb, edge },
      flagged: edge >= EDGE_FLAG_THRESHOLD,
      confidence,
      description,
    };
  }

  // Rescales modelProb so 0 = baseline (the model saying no more than an
  // even split across outcomes would) and 1 = certainty, then buckets it
  // into a label for display.
  private confidenceFor(modelProb: number, baseline: number): Confidence {
    const score = Math.max(0, Math.min(1, (modelProb - baseline) / (1 - baseline)));
    const level: ConfidenceLevel =
      score >= CONFIDENCE_HIGH_MIN ? 'high' : score >= CONFIDENCE_MEDIUM_MIN ? 'medium' : 'low';
    return { level, score };
  }

  // The outcome the double chance leaves out: the one the market most
  // overprices relative to the model, i.e. the minimum signed edge. Without
  // a market, the least likely outcome - covering the two likeliest.
  private doubleChanceExcludedSide(
    probs: Probabilities,
    market: Probabilities | null,
    rule: CallRule,
  ): Outcome {
    // max_prob never consults the market, and without a market max_edge
    // cannot either - both fall back to the model's own ranking, covering
    // the two outcomes it believes in most.
    if (rule === 'max_prob' || market === null) {
      return OUTCOMES.reduce((worst, o) => (probs[o] < probs[worst] ? o : worst));
    }
    const edge = (o: Outcome) => probs[o] - market[o];
    return OUTCOMES.reduce((worst, o) => (edge(o) < edge(worst) ? o : worst));
  }

  // Largest-edge outcome where there's a market to compare against, else the
  // likeliest outcome. Unlike before, this never bails on an unflagged
  // fixture - the flag only decides how the pick is presented.
  private callSide(
    probs: Probabilities,
    market: Probabilities | null,
    rule: CallRule,
  ): Outcome {
    // Same fallback as the double-chance selector: max_prob is defined as
    // the model's own favourite, and a max_edge with no market to compare
    // against has nothing else to go on.
    if (rule === 'max_prob' || market === null) {
      return OUTCOMES.reduce((best, o) => (probs[o] > probs[best] ? o : best));
    }
    const edge = (o: Outcome) => probs[o] - market[o];
    // Signed, not Math.abs: the pick is the outcome the market most
    // UNDERprices. Selecting on magnitude backs an outcome the model thinks
    // is overpriced whenever the biggest disagreement is negative, which on
    // the first v2/v3 backfill was 29 of 43 fixtures. Always names a side -
    // the three edges sum to 0, so at least one is non-negative.
    return OUTCOMES.reduce((best, o) => (edge(o) > edge(best) ? o : best));
  }
}
