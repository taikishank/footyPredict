# Project: LiveEdge — Real-Time Sports Win-Probability & Market Edge Detector

## 1. Overview

A full-stack, cloud-deployed system that:
1. Ingests live/near-live soccer and football match data from a third-party sports API.
2. Runs an ML model to compute real-time win probability / event predictions.
3. Fetches current betting odds and converts them to implied probabilities.
4. Uses the Claude API (with web search enabled) to pull qualitative context (injuries, lineup news, weather) and compare the model's probability against the market's.
5. Surfaces statistical "edges" (model vs. market divergence) on a live web dashboard.
6. Backtests historical edges against actual outcomes to evaluate model calibration.

**Purpose:** Portfolio/resume project demonstrating full-stack engineering, ML, cloud infra, and agentic AI tool use. NOT a betting product — framed as analytical/educational tooling. README must state this explicitly.

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion service | Go |
| ML training & inference | Python (FastAPI, XGBoost/LightGBM, scikit-learn) |
| Context/odds synthesis | Claude API (web search tool enabled) |
| Frontend | TypeScript + Angular (Angular CLI), WebSockets for live updates |
| Queue | AWS SQS (or Kafka if self-managed) |
| Database | AWS DynamoDB (live match state) + RDS Postgres (historical data, backtests) |
| Object storage | AWS S3 (training datasets, model artifacts) |
| Compute/deploy | AWS ECS Fargate (Go ingestor, FastAPI inference), AWS Lambda (scheduled odds/context jobs) |
| API gateway | AWS API Gateway |
| IaC | Terraform (preferred) or AWS CDK |
| CI/CD | GitHub Actions |

---

## 3. Architecture

```
[Sports Data API] --> [Go Ingestion Service] --> [SQS Queue] --> [Match State: DynamoDB]
                                                                          |
[Odds API] --> [Lambda: Odds Fetcher] ---------------------------------->|
                                                                          |
[Claude API + Web Search] --> [Lambda: Context Synthesizer] ------------>|
                                                                          v
                                                        [FastAPI Inference Service]
                                                        (win probability + edge calc)
                                                                          |
                                                                          v
                                                        [Angular Dashboard] <--(WebSocket)--
                                                                          |
                                                                          v
                                                [Postgres: historical edges + outcomes]
                                                                          |
                                                                          v
                                                        [Backtest module / calibration report]
```

---

## 4. Build Phases (sequenced for incremental, demoable progress)

### Phase 0 — Repo & Infra Scaffolding
- Monorepo structure (see §6)
- Terraform skeleton for S3, DynamoDB, RDS, SQS, ECS, Lambda, API Gateway
- GitHub Actions CI: lint + test on push

### Phase 1 — Historical ML Model (offline)
- Pull historical match data (free tier: football-data.org or API-Football)
- Feature engineering: team form, xG, home/away, rest days, head-to-head
- Train XGBoost/LightGBM classifier for match outcome probability
- Evaluate with log loss, Brier score, calibration curve
- Store model artifact in S3

### Phase 2 — Batch Pipeline
- Go service pulls recent completed matches on a schedule
- Writes to Postgres (historical) and triggers feature recompute
- FastAPI endpoint serves predictions for a given match ID

### Phase 3 — Live Layer
- Go service polls live match data during active matches, pushes events to SQS
- DynamoDB holds current match state (score, minute, momentum stats)
- FastAPI computes live win probability, updates on each event
- Angular dashboard shows live win-probability chart via WebSocket

### Phase 4 — Odds & Edge Detection
- Lambda fetches odds from Odds API on a schedule for upcoming/live matches
- Convert odds → implied probability, de-vig
- Compute edge = model_prob − market_implied_prob
- Store edges in Postgres; flag threshold-exceeding edges

### Phase 5 — Claude API Context Synthesis
- Lambda calls Claude API with web search tool enabled
- Prompt: gather injury news, lineup changes, weather for upcoming match
- Claude returns structured qualitative summary + confidence flag
- Merge into dashboard alongside quantitative edge

### Phase 6 — Backtesting & Calibration
- Historical edges vs. actual outcomes stored in Postgres
- Backtest module computes: hit rate on flagged edges, calibration plot, ROI simulation (paper only)
- Render as a report page in the dashboard

### Phase 7 — Polish for Resume/Demo
- README with architecture diagram, setup instructions, screenshots/GIFs
- Deploy live demo (or record a demo video if live odds/cost is a concern)
- Write a short case-study doc: problem, approach, results, what you'd improve

---

## 5. Data Sources (evaluate free tiers first)

- **Match data:** football-data.org (free tier), API-Football (RapidAPI), SportRadar (trial)
- **Odds data:** The Odds API (free tier ~500 req/month)
- **Context/news:** Claude API web search tool (no separate news API needed initially)

---

## 6. Monorepo Structure

```
liveedge/
├── infra/                  # Terraform/CDK IaC
├── services/
│   ├── ingestor-go/        # Go: live/batch data ingestion
│   ├── inference-py/       # FastAPI: ML inference + edge calc
│   ├── odds-lambda/        # Lambda: odds fetch + de-vig
│   └── context-lambda/     # Lambda: Claude API context synthesis
├── ml/
│   ├── notebooks/          # Exploratory training work
│   ├── train.py            # Training pipeline
│   └── models/             # Saved model artifacts (gitignored, S3-backed)
├── frontend/                # Angular + TS dashboard
├── backtest/                 # Backtesting scripts + calibration reports
├── docs/
│   ├── architecture.md
│   └── case-study.md
└── README.md
```

---

## 7. Environment Variables (placeholder — fill in via .env, never commit)

```
SPORTS_DATA_API_KEY=
ODDS_API_KEY=
ANTHROPIC_API_KEY=
AWS_REGION=
DYNAMODB_TABLE_NAME=
POSTGRES_URL=
S3_BUCKET_MODELS=
SQS_QUEUE_URL=
```

---

## 8. Non-Goals / Constraints

- Not a betting execution platform — no real-money wagering integration.
- Keep API costs low during development: cache odds/context calls, use free-tier data where possible, avoid polling live matches outside of demo windows.
- Prioritize a working end-to-end slice (Phase 0–3) before adding odds/Claude layers — a live demo of win-probability alone is already resume-worthy if time runs short.

---

## 9. Instructions for Claude Code

When initializing this repo:
1. Scaffold the monorepo structure in §6.
2. Start with Phase 0 and Phase 1 — get a working historical model before touching live infra.
3. Use idiomatic Go for the ingestion service (modules, proper error handling, structured logging).
4. Use FastAPI + Pydantic for the inference service with typed request/response models.
5. Frontend: standalone Angular components, TypeScript strict mode, minimal state management (Angular signals — avoid over-engineering with NgRx for this scope).
6. Write unit tests alongside each service as it's built, not as an afterthought.
7. Keep Terraform modules small and composable (one module per AWS resource group).
8. Flag any step that requires an external API key or paid tier before proceeding, so the user can supply credentials or choose an alternative.
