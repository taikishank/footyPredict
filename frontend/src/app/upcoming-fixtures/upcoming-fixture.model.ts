// Mirrors inference-py/app/schemas.py's UpcomingFixture/UpcomingFixturesResponse
// - keep in sync manually since there's no shared schema codegen yet.
import type { Probabilities } from '../live-prediction/live-prediction.model';

// Mirrors inference-py/app/schemas.py's BTTSProbabilities - a separate,
// two-outcome market (yes/no), not a variant of Probabilities.
export interface BTTSProbabilities {
  yes: number;
  no: number;
}

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
  // BTTS is a second, independent model/market on the same fixture - all
  // four are null together whenever the BTTS model isn't loaded or the
  // fixture has no v3-shaped feature vector to feed it.
  btts_probabilities: BTTSProbabilities | null;
  btts_market: BTTSProbabilities | null;
  btts_edge: BTTSProbabilities | null;
  btts_flagged: boolean;
}

export interface UpcomingFixturesResponse {
  generated_at: string;
  window_days: number;
  fixtures: UpcomingFixture[];
}
