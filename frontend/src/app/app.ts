import { Component, signal } from '@angular/core';

import { LiveDashboardComponent } from './live-prediction/live-dashboard.component';
import { UpcomingFixturesComponent } from './upcoming-fixtures/upcoming-fixtures.component';

type Tab = 'live' | 'upcoming';

@Component({
  selector: 'app-root',
  imports: [LiveDashboardComponent, UpcomingFixturesComponent],
  templateUrl: './app.html',
})
export class App {
  protected readonly activeTab = signal<Tab>('live');

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
