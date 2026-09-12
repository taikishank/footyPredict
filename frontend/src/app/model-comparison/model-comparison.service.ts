import { Injectable, PLATFORM_ID, inject, signal } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

import { environment } from '../../environments/environment';
import type { ModelComparisonResponse } from './model-comparison.model';

export type FetchStatus = 'idle' | 'loading' | 'loaded' | 'not-available' | 'error';

// Fetches inference-py's GET /models/comparison - the offline validation
// comparison between the incumbent (team form) model and the
// availability-tier challenger, not a live prediction (see
// inference-py/app/model_comparison.py). A one-shot request, same as the
// Upcoming tab, since this only needs to refresh on tab entry.
@Injectable({ providedIn: 'root' })
export class ModelComparisonService {
  private readonly isBrowser = isPlatformBrowser(inject(PLATFORM_ID));

  readonly status = signal<FetchStatus>('idle');
  readonly comparison = signal<ModelComparisonResponse | null>(null);
  readonly errorMessage = signal<string | null>(null);

  async load(): Promise<void> {
    if (!this.isBrowser) return; // no fetch during SSR

    this.status.set('loading');
    this.errorMessage.set(null);

    try {
      const response = await fetch(`${environment.apiBaseUrl}/models/comparison`);
      if (response.status === 404) {
        // No train_challenger.py run has been logged to MLflow yet.
        this.status.set('not-available');
        return;
      }
      if (!response.ok) {
        throw new Error(`request failed with status ${response.status}`);
      }
      const body = (await response.json()) as ModelComparisonResponse;
      this.comparison.set(body);
      this.status.set('loaded');
    } catch (err) {
      this.status.set('error');
      this.errorMessage.set(
        err instanceof Error ? err.message : 'failed to load model comparison',
      );
    }
  }
}
