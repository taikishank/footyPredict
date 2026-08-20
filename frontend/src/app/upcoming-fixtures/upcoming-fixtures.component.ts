import { ChangeDetectionStrategy, Component, OnInit, inject, output } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';

import { UpcomingFixturesService } from './upcoming-fixtures.service';

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
}
