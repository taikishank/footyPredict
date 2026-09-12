import { ChangeDetectionStrategy, Component, OnInit, inject, output } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';

import { OUTCOME_LABEL } from '../live-prediction/live-prediction.model';
import { UpcomingFixturesService } from './upcoming-fixtures.service';
import type { Outcome, UpcomingFixture } from './upcoming-fixture.model';

// Double-chance label for the bet that excludes a given singleton outcome
// (e.g. excluding away_win leaves home-or-draw, i.e. "1X"). Mirrors
// backtest/grade_calls.py's _DC_LABEL.
const DC_LABEL: Record<Outcome, string> = {
  away_win: '1X',
  home_win: 'X2',
  draw: '12',
};

export interface MatchPrediction {
  label: string;
  probability: number;
}

export interface EdgeCall {
  label: string;
  modelProb: number;
  marketProb: number;
  edge: number;
}

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

  // The model's outright call on the match - the outcome it rates most likely.
  // Present for every fixture it could build features for, independent of what
  // the market thinks, so the list always shows a prediction.
  protected prediction(fx: UpcomingFixture): MatchPrediction | null {
    if (fx.prediction === null || !fx.probabilities) return null;

    return {
      label: OUTCOME_LABEL[fx.prediction],
      probability: fx.probabilities[fx.prediction],
    };
  }

  // The betting call on a flagged fixture: whichever outcome diverges most from
  // the market. Mirrors backtest/grade_calls.py's call_side selection so the
  // value call shown here means the same thing it means once graded against the
  // actual result. Distinct from prediction() above - this one needs odds.
  protected edgeCall(fx: UpcomingFixture): EdgeCall | null {
    const side = this.callSide(fx);
    if (side === null) return null;

    return {
      label: OUTCOME_LABEL[side],
      modelProb: fx.probabilities![side],
      marketProb: fx.market![side],
      edge: fx.edge![side],
    };
  }

  // The same call reframed as a double-chance bet: the pair of outcomes
  // excluding whichever side edgeCall() called. Derived from the same
  // probabilities/market/edge already on the fixture - mirrors
  // backtest/grade_calls.py's double_chance_calls derivation.
  protected edgeCallDoubleChance(fx: UpcomingFixture): EdgeCall | null {
    const side = this.callSide(fx);
    if (side === null) return null;

    return {
      label: DC_LABEL[side],
      modelProb: 1 - fx.probabilities![side],
      marketProb: 1 - fx.market![side],
      edge: (1 - fx.probabilities![side]) - (1 - fx.market![side]),
    };
  }

  private callSide(fx: UpcomingFixture): Outcome | null {
    if (!fx.flagged || !fx.probabilities || !fx.market || !fx.edge) return null;

    const outcomes: Outcome[] = ['home_win', 'draw', 'away_win'];
    return outcomes.reduce((best, o) =>
      Math.abs(fx.edge![o]) > Math.abs(fx.edge![best]) ? o : best,
    );
  }
}
