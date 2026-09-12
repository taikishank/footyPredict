import { ChangeDetectionStrategy, Component, OnInit, inject } from '@angular/core';
import { DecimalPipe } from '@angular/common';

import { ModelComparisonService } from './model-comparison.service';

@Component({
  selector: 'app-model-comparison',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [DecimalPipe],
  templateUrl: './model-comparison.component.html',
})
export class ModelComparisonComponent implements OnInit {
  protected readonly comparisonService = inject(ModelComparisonService);

  ngOnInit(): void {
    void this.comparisonService.load();
  }
}
