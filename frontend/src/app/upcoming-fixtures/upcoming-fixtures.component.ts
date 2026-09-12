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
}

// Mirrors inference-py/app/config.py's EDGE_FLAG_THRESHOLD default, for the
// picks whose flag isn't computed server-side: only the incumbent's
// edge/flagged are persisted to `edges` for backtest/grade_calls.py, so the
// v2 challenger's is derived client-side against the same gate.
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

  // The side the model is calling, for every fixture it could predict at
  // all - not just the flagged ones. Where there's a market, that's
  // whichever outcome has the largest edge magnitude, mirroring
  // backtest/grade_calls.py's call_side selection so "the model's pick"
  // here means the same thing it means once the call gets graded against
  // the actual result; without a market it's simply the likeliest outcome.
  protected modelPick(fx: UpcomingFixture): ModelPick | null {
    // The incumbent's flag comes from the server, which owns
    // EDGE_FLAG_THRESHOLD and persists the snapshot the backtest grades.
    return this.buildPick(fx.probabilities, fx.market, OUTCOME_LABEL, fx.flagged);
  }

  // The same call reframed as a double-chance bet: the pair of outcomes
  // excluding whichever side modelPick() called. Mirrors
  // backtest/grade_calls.py's double_chance_calls derivation.
  protected modelPickDoubleChance(fx: UpcomingFixture): ModelPick | null {
    return this.buildPick(fx.probabilities, fx.market, DC_LABEL, fx.flagged, true);
  }

  // Same shape as modelPick()/modelPickDoubleChance(), for the v2 challenger.
  protected modelPickV2(fx: UpcomingFixture): ModelPick | null {
    return this.buildPick(fx.probabilities_v2, fx.market, OUTCOME_LABEL);
  }

  protected modelPickDoubleChanceV2(fx: UpcomingFixture): ModelPick | null {
    return this.buildPick(fx.probabilities_v2, fx.market, DC_LABEL, undefined, true);
  }

  /**
   * Builds the pick for one model's probabilities. `flaggedOverride` lets the
   * incumbent defer to the server's flag; otherwise the edge is gated
   * client-side. `doubleChance` complements the called side into the
   * two-outcome bet, which leaves the flag untouched - it's the same call.
   */
  private buildPick(
    probs: Probabilities | null,
    market: Probabilities | null,
    labels: Record<Outcome, string>,
    flaggedOverride?: boolean,
    doubleChance = false,
  ): ModelPick | null {
    if (!probs) return null;

    const side = this.callSide(probs, market);
    const modelProb = doubleChance ? 1 - probs[side] : probs[side];

    if (market === null) {
      return { label: labels[side], modelProb, vsMarket: null, flagged: flaggedOverride ?? false };
    }

    const marketProb = doubleChance ? 1 - market[side] : market[side];
    // The flag is always judged on the singleton edge, so the double-chance
    // framing of a call carries the same flag as the call itself.
    const singletonEdge = probs[side] - market[side];

    return {
      label: labels[side],
      modelProb,
      vsMarket: { marketProb, edge: modelProb - marketProb },
      flagged: flaggedOverride ?? Math.abs(singletonEdge) >= EDGE_FLAG_THRESHOLD,
    };
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
