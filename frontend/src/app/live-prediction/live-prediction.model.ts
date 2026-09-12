// Mirrors inference-py/app/schemas.py's LivePredictionResponse - keep in
// sync manually since there's no shared schema codegen yet.
export interface Probabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export type Outcome = keyof Probabilities;

// Display names for the three outcomes, shared by the Upcoming and Live views.
export const OUTCOME_LABEL: Record<Outcome, string> = {
  home_win: 'Home win',
  draw: 'Draw',
  away_win: 'Away win',
};

export interface LivePrediction {
  fixture_id: number;
  league_id: number;
  starting_at: string;
  home_team: string;
  away_team: string;
  probabilities: Probabilities;
  // The outcome the model currently calls, i.e. the most likely of the three
  // once the live score is folded in.
  prediction: Outcome;
  model_version: string;
  low_confidence: boolean;
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
