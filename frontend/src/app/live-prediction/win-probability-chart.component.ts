import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';

import type { ProbabilityPoint } from './live-prediction.model';

interface Series {
  label: string;
  strokeClass: string;
  dotClass: string;
  points: string;
  latestPct: number;
}

const VIEW_WIDTH = 100;
const VIEW_HEIGHT = 100;

@Component({
  selector: 'app-win-probability-chart',
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './win-probability-chart.component.html',
})
export class WinProbabilityChartComponent {
  readonly points = input<ProbabilityPoint[]>([]);

  protected readonly series = computed<Series[]>(() => {
    const data = this.points();
    return [
      { key: 'home_win' as const, label: 'Home win', strokeClass: 'stroke-emerald-500', dotClass: 'bg-emerald-500' },
      { key: 'draw' as const, label: 'Draw', strokeClass: 'stroke-slate-400', dotClass: 'bg-slate-400' },
      { key: 'away_win' as const, label: 'Away win', strokeClass: 'stroke-rose-500', dotClass: 'bg-rose-500' },
    ].map(({ key, label, strokeClass, dotClass }) => ({
      label,
      strokeClass,
      dotClass,
      points: toPolylinePoints(data, key),
      latestPct: Math.round((data.at(-1)?.probabilities[key] ?? 0) * 100),
    }));
  });
}

function toPolylinePoints(data: ProbabilityPoint[], key: keyof ProbabilityPoint['probabilities']): string {
  if (data.length === 0) return '';
  if (data.length === 1) {
    const y = VIEW_HEIGHT * (1 - data[0].probabilities[key]);
    return `0,${y} ${VIEW_WIDTH},${y}`;
  }
  const step = VIEW_WIDTH / (data.length - 1);
  return data
    .map((point, i) => `${i * step},${VIEW_HEIGHT * (1 - point.probabilities[key])}`)
    .join(' ');
}
