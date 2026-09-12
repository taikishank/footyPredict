// Mirrors inference-py/app/schemas.py's ModelPerformance/ModelComparisonResponse
// - keep in sync manually since there's no shared schema codegen yet.
export interface ModelPerformance {
  label: string;
  log_loss: number;
  brier: number;
}

export interface ModelComparisonResponse {
  run_id: string;
  n_train_rows: number;
  n_val_rows: number;
  incumbent: ModelPerformance;
  challenger: ModelPerformance;
  log_loss_improvement: number;
  ci_low: number;
  ci_high: number;
  p_better: number;
  clears_promotion_bar: boolean;
}
