// Mirrors inference-py/app/schemas.py's UpcomingFixture/UpcomingFixturesResponse
// - keep in sync manually since there's no shared schema codegen yet.
import type { Probabilities } from '../live-prediction/live-prediction.model';

export interface UpcomingFixture {
  fixture_id: number;
  league_id: number;
  starting_at: string;
  home_team: string;
  away_team: string;
  probabilities: Probabilities | null;
  model_version: string | null;
}

export interface UpcomingFixturesResponse {
  generated_at: string;
  window_days: number;
  fixtures: UpcomingFixture[];
}
