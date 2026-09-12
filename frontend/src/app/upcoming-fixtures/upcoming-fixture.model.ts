// Mirrors inference-py/app/schemas.py's UpcomingFixture/UpcomingFixturesResponse
// - keep in sync manually since there's no shared schema codegen yet.
import type { Probabilities } from '../live-prediction/live-prediction.model';

export interface UpcomingFixture {
  fixture_id: number;
  league_id: number;
  starting_at: string;
  home_team: string;
  away_team: string;
  // The incumbent's (v1) prediction. No longer shown in the UI now that
  // backtesting confirmed v2 is the better model, but still served so its
  // season-long stats keep getting tracked (see backtest/grade_calls.py).
  probabilities: Probabilities | null;
  model_version: string | null;
  market: Probabilities | null;
  edge: Probabilities | null;
  flagged: boolean;
  // The v2 model's prediction (team form + club_goals_relative) - null under
  // the same conditions as `probabilities` (not enough history, or the v2
  // model isn't loaded). This is the prediction shown in the UI.
  probabilities_v2: Probabilities | null;
  model_version_v2: string | null;
}

export interface UpcomingFixturesResponse {
  generated_at: string;
  window_days: number;
  fixtures: UpcomingFixture[];
}
