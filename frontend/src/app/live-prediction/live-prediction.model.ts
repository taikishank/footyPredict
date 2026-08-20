// Mirrors inference-py/app/schemas.py's LivePredictionResponse - keep in
// sync manually since there's no shared schema codegen yet.
export interface Probabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface LivePrediction {
  fixture_id: number;
  league_id: number;
  starting_at: string;
  home_team: string;
  away_team: string;
  probabilities: Probabilities;
  model_version: string;
  state: string;
  home_goals: number;
  away_goals: number;
  live_state_updated_at: string;
}

export interface ProbabilityPoint {
  receivedAt: number;
  probabilities: Probabilities;
  minuteLabel: string;
}

export type ConnectionStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error';
