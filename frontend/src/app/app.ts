import { Component, computed, inject, signal } from '@angular/core';

import { LiveDashboardComponent } from './live-prediction/live-dashboard.component';
import { LivePredictionService } from './live-prediction/live-prediction.service';
import { ModelComparisonComponent } from './model-comparison/model-comparison.component';
import { UpcomingFixturesComponent } from './upcoming-fixtures/upcoming-fixtures.component';
import { UpcomingFixturesService } from './upcoming-fixtures/upcoming-fixtures.service';

type Tab = 'live' | 'upcoming' | 'compare';

@Component({
  selector: 'app-root',
  imports: [LiveDashboardComponent, UpcomingFixturesComponent, ModelComparisonComponent],
  templateUrl: './app.html',
})
export class App {
  private readonly live = inject(LivePredictionService);
  private readonly upcoming = inject(UpcomingFixturesService);

  protected readonly activeTab = signal<Tab>('live');

  // Shown in the footer on both tabs; falls back from whichever source has
  // reported a model version most recently (live stream, then upcoming list).
  protected readonly modelVersion = computed(
    () => this.live.latest()?.model_version ?? this.upcoming.fixtures().find((f) => f.model_version)?.model_version ?? null,
  );

  // Set by the Upcoming tab's "Watch live" handoff (PROJECT_SPEC.md Phase 4).
  protected readonly handoffFixtureId = signal<number | null>(null);

  protected selectTab(tab: Tab): void {
    this.activeTab.set(tab);
  }

  protected watchOnLiveTab(fixtureId: number): void {
    this.handoffFixtureId.set(fixtureId);
    this.activeTab.set('live');
  }
}
