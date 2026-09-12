import { ChangeDetectionStrategy, Component, OnInit, inject, output } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';

import { UpcomingFixturesService } from './upcoming-fixtures.service';
import type { UpcomingFixture } from './upcoming-fixture.model';

type Outcome = 'home_win' | 'draw' | 'away_win';

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

  // The side the model is calling for a flagged fixture: whichever outcome
  // has the largest edge magnitude. Mirrors backtest/grade_calls.py's
  // call_side selection so "the model's pick" here means the same thing it
  // means once the call gets graded against the actual result.
  protected modelPick(fx: UpcomingFixture): ModelPick | null {
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
  // excluding whichever side modelPick() called. Derived from the same
  // probabilities/market/edge already on the fixture - mirrors
  // backtest/grade_calls.py's double_chance_calls derivation.
  protected modelPickDoubleChance(fx: UpcomingFixture): ModelPick | null {
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
