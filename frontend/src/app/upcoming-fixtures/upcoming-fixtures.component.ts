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
// backtest/grade_calls.py's _DC_LABEL.
const DC_LABEL: Record<Outcome, string> = {
  away_win: '1X',
  home_win: 'X2',
  draw: '12',
};

export type ConfidenceLevel = 'low' | 'medium' | 'high';

export interface Confidence {
  level: ConfidenceLevel;
  // 0-1: how far modelProb sits above the no-information baseline for this
  // bet type, rescaled so 0 = baseline (a coin-flip-equivalent call) and
  // 1 = certainty. Distinct from modelProb itself, which is an absolute
  // probability, not a measure of how much the model is telling you.
  score: number;
}

export interface ModelPick {
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
}

// Score thresholds for bucketing a continuous confidence score into a label.
const CONFIDENCE_MEDIUM_MIN = 0.3;
const CONFIDENCE_HIGH_MIN = 0.65;

// Mirrors inference-py/app/config.py's EDGE_FLAG_THRESHOLD default. The
// server persists fx.flagged for the incumbent (v1) model, which backtest
// grade_calls.py grades, but the UI now shows the v2 challenger's pick, so
// its flag is derived client-side against the same gate.
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

  // The side the model is calling, for every fixture it could predict at
  // all - not just the flagged ones. Where there's a market, that's
  // whichever outcome has the largest edge magnitude, mirroring
  // backtest/grade_calls.py's call_side selection so "the model's pick"
  // here means the same thing it means once the call gets graded against
  // the actual result; without a market it's simply the likeliest outcome.
  // Uses the v2 challenger's probabilities - it's replaced the incumbent as
  // the model shown here now that backtesting confirmed it's the better
  // model; the incumbent (fx.probabilities) is still served for tracking,
  // just not surfaced in this UI.
  protected modelPick(fx: UpcomingFixture): ModelPick | null {
    return this.buildPick(fx.probabilities_v2, fx.market, OUTCOME_LABEL);
  }

  // The same call reframed as a double-chance bet: the pair of outcomes
  // excluding whichever side modelPick() called. Mirrors
  // backtest/grade_calls.py's double_chance_calls derivation.
  protected modelPickDoubleChance(fx: UpcomingFixture): ModelPick | null {
    return this.buildPick(fx.probabilities_v2, fx.market, DC_LABEL, true);
  }

  /**
   * Builds the pick for one model's probabilities. The edge is always gated
   * client-side against EDGE_FLAG_THRESHOLD. `doubleChance` complements the
   * called side into the two-outcome bet, which leaves the flag untouched -
   * it's the same call.
   */
  private buildPick(
    probs: Probabilities | null,
    market: Probabilities | null,
    labels: Record<Outcome, string>,
    doubleChance = false,
  ): ModelPick | null {
    if (!probs) return null;

    const side = this.callSide(probs, market);
    const modelProb = doubleChance ? 1 - probs[side] : probs[side];
    // A double-chance bet covers 2 of the 3 outcomes, so a "no information"
    // model would still assign it 2/3; a singleton call's baseline is 1/3.
    const baseline = doubleChance ? 2 / 3 : 1 / 3;
    const confidence = this.confidenceFor(modelProb, baseline);

    if (market === null) {
      return {
        label: labels[side],
        modelProb,
        vsMarket: null,
        flagged: false,
        confidence,
      };
    }

    const marketProb = doubleChance ? 1 - market[side] : market[side];
    // The flag is always judged on the singleton edge, so the double-chance
    // framing of a call carries the same flag as the call itself. Only a
    // positive edge (model sees more value than the market prices in) is
    // worth flagging - a large negative edge just means the market likes
    // that outcome more than the model does, not a call worth highlighting.
    const singletonEdge = probs[side] - market[side];

    return {
      label: labels[side],
      modelProb,
      vsMarket: { marketProb, edge: modelProb - marketProb },
      flagged: singletonEdge >= EDGE_FLAG_THRESHOLD,
      confidence,
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

  // Largest-edge outcome where there's a market to compare against, else the
  // likeliest outcome. Unlike before, this never bails on an unflagged
  // fixture - the flag only decides how the pick is presented.
  private callSide(probs: Probabilities, market: Probabilities | null): Outcome {
    if (market === null) {
      return OUTCOMES.reduce((best, o) => (probs[o] > probs[best] ? o : best));
    }
    const edge = (o: Outcome) => probs[o] - market[o];
    return OUTCOMES.reduce((best, o) => (Math.abs(edge(o)) > Math.abs(edge(best)) ? o : best));
  }
}
