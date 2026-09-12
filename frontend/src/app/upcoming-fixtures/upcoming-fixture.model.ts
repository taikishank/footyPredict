// Mirrors inference-py/app/schemas.py's UpcomingFixture/UpcomingFixturesResponse
// - keep in sync manually since there's no shared schema codegen yet.
import type { Outcome, Probabilities } from '../live-prediction/live-prediction.model';

export type { Outcome };

export interface UpcomingFixture {
  fixture_id: number;
  league_id: number;
  starting_at: string;
  home_team: string;
  away_team: string;
  probabilities: Probabilities | null;
  // The outcome the model calls - present for every fixture it could build
  // features for, whether or not the market disagrees.
  prediction: Outcome | null;
  model_version: string | null;
  // At least one team has thin form data this season, so the prediction rests
  // on fewer matches than the model was trained on.
  low_confidence: boolean;
  market: Probabilities | null;
  edge: Probabilities | null;
  flagged: boolean;
}

export interface UpcomingFixturesResponse {
  generated_at: string;
  window_days: number;
  fixtures: UpcomingFixture[];
}
