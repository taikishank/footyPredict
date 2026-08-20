// Dev-only: Phase 3 is a local demo (see PROJECT_SPEC.md), inference-py
// runs on localhost:8000. Swap for a real env-based config once this is
// actually deployed behind API Gateway.
export const environment = {
  wsBaseUrl: 'ws://localhost:8000',
  apiBaseUrl: 'http://localhost:8000',
};
