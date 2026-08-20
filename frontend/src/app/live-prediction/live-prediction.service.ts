import { Injectable, PLATFORM_ID, inject, signal } from '@angular/core';
import { isPlatformBrowser } from '@angular/common';

import { environment } from '../../environments/environment';
import type { ConnectionStatus, LivePrediction, ProbabilityPoint } from './live-prediction.model';

const MAX_HISTORY_POINTS = 200;

// Owns the WebSocket connection to inference-py's
// /ws/predictions/{fixture_id}/live route and exposes the stream as signals.
// Kept as one plain injectable rather than NgRx, per PROJECT_SPEC.md §9.5.
@Injectable({ providedIn: 'root' })
export class LivePredictionService {
  private readonly isBrowser = isPlatformBrowser(inject(PLATFORM_ID));
  private socket: WebSocket | null = null;

  readonly status = signal<ConnectionStatus>('idle');
  readonly latest = signal<LivePrediction | null>(null);
  readonly history = signal<ProbabilityPoint[]>([]);
  readonly errorMessage = signal<string | null>(null);

  connect(fixtureId: number): void {
    if (!this.isBrowser) return; // no WebSocket on the server during SSR

    this.disconnect();
    this.history.set([]);
    this.errorMessage.set(null);
    this.status.set('connecting');

    const socket = new WebSocket(`${environment.wsBaseUrl}/ws/predictions/${fixtureId}/live`);
    this.socket = socket;

    socket.onopen = () => this.status.set('open');

    socket.onmessage = (event) => {
      const prediction = JSON.parse(event.data) as LivePrediction;
      this.latest.set(prediction);
      this.history.update((points) => {
        const next: ProbabilityPoint[] = [
          ...points,
          {
            receivedAt: Date.now(),
            probabilities: prediction.probabilities,
            minuteLabel: prediction.state,
          },
        ];
        return next.length > MAX_HISTORY_POINTS ? next.slice(-MAX_HISTORY_POINTS) : next;
      });
    };

    socket.onerror = () => {
      this.status.set('error');
      this.errorMessage.set('WebSocket error - check the fixture id and that inference-py is running.');
    };

    socket.onclose = (event) => {
      // Server closes with 4004 when the fixture doesn't exist (see main.py).
      if (event.code === 4004) {
        this.errorMessage.set(event.reason || `fixture not found`);
      }
      this.status.set('closed');
      this.socket = null;
    };
  }

  disconnect(): void {
    this.socket?.close();
    this.socket = null;
    if (this.status() !== 'error') this.status.set('idle');
  }
}
