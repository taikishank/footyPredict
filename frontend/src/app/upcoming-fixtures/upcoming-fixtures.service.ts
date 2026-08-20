import { Injectable, PLATFORM_ID, inject, signal } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

import { environment } from '../../environments/environment';
import type { UpcomingFixture, UpcomingFixturesResponse } from './upcoming-fixture.model';

export type FetchStatus = 'idle' | 'loading' | 'loaded' | 'error';

// Fetches inference-py's GET /fixtures/upcoming - a one-shot request rather
// than a WebSocket, since this list only needs to refresh on tab entry, not
// stream (see PROJECT_SPEC.md Phase 4's Upcoming tab).
@Injectable({ providedIn: 'root' })
export class UpcomingFixturesService {
  private readonly isBrowser = isPlatformBrowser(inject(PLATFORM_ID));

  readonly status = signal<FetchStatus>('idle');
  readonly fixtures = signal<UpcomingFixture[]>([]);
  readonly errorMessage = signal<string | null>(null);

  async load(): Promise<void> {
    if (!this.isBrowser) return; // no fetch during SSR

    this.status.set('loading');
    this.errorMessage.set(null);

    try {
      const response = await fetch(`${environment.apiBaseUrl}/fixtures/upcoming`);
      if (!response.ok) {
        throw new Error(`request failed with status ${response.status}`);
      }
      const body = (await response.json()) as UpcomingFixturesResponse;
      this.fixtures.set(body.fixtures);
      this.status.set('loaded');
    } catch (err) {
      this.status.set('error');
      this.errorMessage.set(
        err instanceof Error ? err.message : 'failed to load upcoming fixtures',
      );
    }
  }
}
