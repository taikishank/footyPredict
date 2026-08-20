import { ChangeDetectionStrategy, Component, effect, inject, input, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import { LivePredictionService } from './live-prediction.service';
import { WinProbabilityChartComponent } from './win-probability-chart.component';

@Component({
  selector: 'app-live-dashboard',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, WinProbabilityChartComponent],
  templateUrl: './live-dashboard.component.html',
})
export class LiveDashboardComponent {
  protected readonly live = inject(LivePredictionService);

  // Set when the Upcoming tab hands off a fixture id (see
  // PROJECT_SPEC.md Phase 4); auto-connects to it once set.
  readonly initialFixtureId = input<number | null>(null);

  protected readonly fixtureIdInput = signal<number | null>(null);

  constructor() {
    effect(() => {
      const fixtureId = this.initialFixtureId();
      if (fixtureId !== null) {
        this.fixtureIdInput.set(fixtureId);
        this.live.connect(fixtureId);
      }
    });
  }

  protected watch(): void {
    const fixtureId = this.fixtureIdInput();
    if (fixtureId !== null) this.live.connect(fixtureId);
  }

  protected stop(): void {
    this.live.disconnect();
  }
}
