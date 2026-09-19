# Goal-Scoring / Team-Strength Model Architecture for v4

Scope: team-level goal/outcome modeling only (Dixon-Coles / bivariate Poisson /
rating systems). Player-level features are covered by another researcher.
Current baseline: v3 (depth-2 XGBoost on 63 columns, `AttackDefence` ratings,
log-pooled with a properly MLE-fit Dixon-Coles score matrix) at **log loss
0.98121** (rolling-origin, 19,797 fixtures), against a de-vigged Pinnacle
closing-line floor of 0.96293 (gap = 0.0183) and class priors at 1.07067–1.07261.

---

## Part A — What this project has already tried (internal findings)

### A1. Goal-scoring model shape (`ml/FINDINGS_GOALS.md`, `ml/goals_lib.py`)

- **Independent Poisson vs Dixon-Coles (τ correction) vs classifier.** Standalone,
  on rolling-origin: independent Poisson 0.98758, Dixon-Coles (ρ≈−0.08) 0.98673,
  served v3 classifier 0.98497, Pinnacle 0.96293. The τ/ρ correlation correction
  is real but small: **+0.00085 standalone** (~4% of the market gap). — internal,
  `ml/FINDINGS_GOALS.md:44-60`
- **"The gain is the fit, not the shape."** The single largest lever found in
  this line of work was *not* changing the Poisson/Dixon-Coles functional form,
  but replacing an online per-fixture gradient-step rate estimate with a proper
  MLE batch fit of the Dixon-Coles likelihood (half-life-weighted, refit on a
  short cadence). That alone bought **+0.00323** over the same score-matrix
  shape driven by online rates (0.98673 → 0.98350 standalone) — 15% of the
  market gap and the largest single movement since the ratings phase. Best
  config: half-life 365 days, refit every 7 days; a 90-day refit cadence
  erases the entire gain (0.98670, indistinguishable from the online rate).
  — internal, `ml/FINDINGS_GOALS.md:107-147`
- **Two Poisson rates ≈ the whole 63-column model.** `dc_h365_c7` standalone
  (0.98350) is statistically indistinguishable from v3's full 63-column
  gradient-boosted classifier (0.98497), P(dc better)=0.929, CI spans zero.
  This is read as "parity, not superiority," and it holds across all
  quarters of the rolling folds (not a refit-recency artifact). — internal,
  `ml/FINDINGS_GOALS.md:150-177`
- **Feeding the score-matrix output to the tree vs. blending outside it.**
  Column arms (`dc_rates`, `dc_probs`, `dc_full` — handing the Poisson rates or
  score-matrix probabilities to the XGBoost tree as extra columns) bought only
  +0.00098 to +0.00137 over v3. **Log-pooling the two forecasters' probability
  outputs outside the tree** (one fitted weight per fold, mean weight ≈0.48–0.57)
  bought **+0.00341** — 2.5× the best column arm and the single largest
  standing finding in the project (16% of the market gap). "The tree is the
  wrong place to put a second forecaster." — internal, `ml/FINDINGS_GOALS.md:248-282`
- **Handing the tree the score-matrix *probabilities* over the raw *rates*
  bought nothing** (+0.00023, CI spans zero) — everything came from having a
  well-fit rate, not from the score-matrix geometry itself. — internal,
  `ml/FINDINGS_GOALS.md:283-294`
- **The draw hypothesis — the original motivation for building a goals model —
  failed its own pre-registered gate, twice** (online rates: draw separation
  +0.0102 vs v3's +0.0119; batch-fit rates: +0.0113, still short of +0.0119).
  A score matrix does *not* pick draws better than the classifier at this fit
  quality, even though draws are 35.5% of the remaining market gap. The blend
  moves the draw column the most of anything measured (1.38950→1.37232) but
  *because two disagreeing forecasters produce a sharper log-pool*, not because
  the Dixon-Coles diagonal mechanism worked as designed. — internal,
  `ml/FINDINGS_GOALS.md:9-14, 142-147, 329-340`
- **A noise-column placebo (shuffled Poisson rates) cleared statistical
  significance twice**, independently of a similar random-interaction placebo
  in `FINDINGS_LEAGUE.md`. This was later (Phase: `FINDINGS_CAPACITY.md`)
  diagnosed as the served depth-4 tree being under-regularized enough that
  noise columns absorbed splits real columns would otherwise waste — i.e. an
  accidental `colsample_bytree` effect, not information. It disappears
  entirely once the tree is retuned to depth 2. — internal,
  `ml/FINDINGS_GOALS.md:296-315`, `ml/FINDINGS_CAPACITY.md:84-108`
- **Open/half-explored items still on the table**: (1) staleness/refit cadence
  has never been measured *for v3 itself* (only for the 2-parameter score
  model) — cheap to check since it was cheap for the score model; (2) the blend
  weight declines across folds (0.58→0.38) suggesting it should be refitted
  online rather than frozen; (3) "two rates match sixty-three columns" implies
  most of the 54 form columns may be redundant once `atkdef` + Dixon-Coles
  rates exist — never tested directly (a `dc_swap` arm replacing `atkdef` with
  the score model, at 61 columns, was a null against v3, but the form-table
  redundancy hypothesis itself was not isolated). — internal, `ml/FINDINGS_GOALS.md:344-395`

### A2. Team ratings (`ml/FINDINGS_RATINGS.md`, `ml/ratings_lib.py`)

- **Attack/defence (Dixon-Coles-style two-rate) beats plain Elo and
  goal-difference Elo decisively**, and is the single largest feature effect
  ever measured in this project: **+0.0076** (rolling), vs. plain Elo +0.0059,
  vs. GD-Elo +0.0058, vs. v2's entire regularization sweep +0.0044. Paired
  directly, attack/defence beats plain Elo by +0.0017 (P=0.999) and GD-Elo by
  +0.0019 (P=1.000). — internal, `ml/FINDINGS_RATINGS.md:33-96`
- **Margin-of-victory scaling in the Elo update is worth nothing** (folding
  goal difference into an Elo K-factor update ties plain win/draw/loss Elo).
  **Stacking all three rating families (Elo, GD-Elo, attack/defence) together
  adds nothing over attack/defence alone** — they are collinear (r≥0.955 on
  the difference columns). — internal, `ml/FINDINGS_RATINGS.md:99-124`
- **Four real, measured biases were found and fixed** in the rating
  implementation, all invisible to log loss and only caught by correlating
  emitted columns against fixture date: (1) pool leakage from seeding
  newcomers below the league mean without a season re-centering; (2) an
  accumulating-cohort bug in league-mean tracking; (3) home-advantage burn-in
  from fitting HFA online from a 0.0 start; (4) `recenter()` not being
  rate-neutral for two-parameter (attack/defence) ratings. — internal,
  `ml/FINDINGS_RATINGS.md:183-220`
- **Archive constraints that any v4 design must respect**: five leagues, zero
  cross-league fixtures, zero clubs playing in more than one league (ratings
  are only ever comparable within a league); 116/211 clubs have a gap season
  (promotion/relegation regression is per-season-elapsed, not a fixed
  discount); newcomers are empirically weaker than the league mean (1.171 vs
  1.362 PPG for incumbents) and are seeded below the mean, not at it.
  — internal, `ml/FINDINGS_RATINGS.md:159-180`
- **Open, unresolved**: serving-shape decision was later settled in favor of
  in-process replay (see A3); hyperparameters were never independently retuned
  for the rated feature set until `FINDINGS_CAPACITY.md`; `PROMOTED_OFFSET`,
  `SEASON_REGRESSION`, and `K` are set by convention/measured sign, never
  swept as a grid.

### A3. Capacity / regularization interaction with ratings (`ml/FINDINGS_CAPACITY.md`)

- **The served v3 tree was over-fitting at `max_depth=4`.** Log loss is
  monotone *increasing* with depth (3→8), the opposite of the "model needs
  more capacity" hypothesis that motivated the phase. Retuning to
  `max_depth=2, reg_lambda=30, n_estimators=800` bought **+0.00191** over the
  served recipe. — internal, `ml/FINDINGS_CAPACITY.md:46-131`
- **Both noise-column placebos (shuffled-rate and random-partition) die to
  zero/negative once depth is fixed at 2**, confirming they were implicit
  regularization artifacts of an over-capacity depth-4 tree, not information.
  — internal, `ml/FINDINGS_CAPACITY.md:84-108`
- **The Dixon-Coles blend "halves" once the tree is properly retuned**:
  +0.00341 at depth 4 becomes +0.00184 at depth 2, because roughly half of the
  originally credited blend gain was the fitted pooling weight compensating
  for a weak, over-fit first forecaster. **It still survives as the largest
  live finding on the board** at depth 2. — internal,
  `ml/FINDINGS_CAPACITY.md:139-160`
- **Same-recipe reseeding alone produces an SD of 0.00006–0.00007 in pooled
  log loss, with 11% of same-recipe seed pairs clearing the project's own 95%
  significance bar** — i.e. that fraction of "significant" findings at small
  effect sizes are expected false positives by construction. **Any v4 gain
  under ~0.0002 should not be trusted without a seed-band check.** — internal,
  `ml/FINDINGS_CAPACITY.md:68-83`
- Methodological point directly relevant to any new architecture: **rating
  state has train/serve equivalence risk.** A stable-vs-unstable sort order
  for same-timestamp fixtures (68% of fixtures share a kickoff instant) moved
  `exp_goal_diff` by up to 1.62 on individual fixtures (SD 0.66) even though
  the *average* effect was inside noise — an asymmetric bug invisible to
  pooled metrics that would have mispriced individual fixtures in production
  indefinitely. Any v4 design that adds more sequential/stateful rating logic
  inherits this risk and needs the same `(date, fixture_id)` stable-sort
  discipline. — internal, `ml/FINDINGS_CAPACITY.md:206-229`

### A4. Recency-weighted index / RWI (`ml/FINDINGS_RWI.md`)

An external Reddit/LinkedIn model ("RWI," reporting 1X2 log loss 0.9797) was
evaluated ingredient-by-ingredient.

- **The headline number is not comparable.** RWI's own reported number *loses
  to its own bookmaker baseline* on log loss, accuracy, and RPS (0.9797 vs.
  0.9779). The comparable quantity — gap-to-market — is −0.0018 for RWI vs.
  **−0.0183** for this project's v3 blend, i.e. this project is already ~10×
  further ahead of its own (much sharper, de-vigged Pinnacle) baseline than
  RWI is of its own (unspecified, likely softer) baseline. — internal,
  `ml/FINDINGS_RWI.md:45-96`
- **5 of RWI's 8 stated ingredients were things this project had already
  ablated and buried**: four-state (home/away × attack/defence) split ratings
  (measured null-to-worse, −0.00102), league-specific scoring prior (measured
  zero), home-advantage prior (dead at depth 2), expected-goals-in-log-space
  score matrix (already the +0.00184 blend, already the largest live finding),
  and per-player lineup-value adjustment (already measured at +0.0061 for
  presence/rotation, +0.0007 for per-player importance weighting).
  — internal, `ml/FINDINGS_RWI.md:100-131`
- **Two novel ingredients were tested and both are null**: (a) *coupled venue
  states* (a partial-coupling axis between fully-shared and fully-split
  attack/defence ratings) — the whole coupling axis from 0 to 1 sits at or
  below the served (fully-coupled) model; a shallow interior optimum at
  coupling=0.5 is still *negative* against v3 (−0.00008). (b) *Bounding the
  per-match residual before it updates the rating* (tanh/Huber clamp on the
  Poisson surprise) — a shallow interior optimum at scale 4 of only +0.00022,
  inside the seed-noise band, and the aggressive end of the bound actively
  hurts away-win prediction specifically (the model's single worst-predicted
  outcome region). — internal, `ml/FINDINGS_RWI.md:318-400`
- **A third RWI ingredient — blending observed goals with a shot-derived (xG
  proxy) rate for the rating update target — was never built.** The project's
  own archive has essentially zero shot-stat coverage before 2015 (0% through
  2005, 1-7% through 2014, jumping to 62.6%→100% by 2016). Because ratings are
  *sequential state replayed from 2000*, a discontinuous data-quality change
  mid-archive would poison every subsequent fixture's rating — this triggered
  a pre-registered hard stop rather than a partial build. **This is a directly
  relevant, well-reasoned rejection of any xG-as-rating-input idea unless a
  full-archive-backfilled or full-history xG source is acquired.** — internal,
  `ml/FINDINGS_RWI.md:284-315`
- **Ordered logit was explicitly considered and rejected on architectural
  grounds**, not tested: an ordered model collapses draw onto one end of a
  single latent strength axis, which is exactly the assumption the
  Dixon-Coles score-matrix route was built to escape (draws are structural,
  from two independent rates, not a band on a line). — internal,
  `ml/FINDINGS_RWI.md:192-199`

### A5. Venue / home-advantage splitting (`ml/FINDINGS_VENUE.md`)

- **Club-level home-advantage/venue strength does not exist as a measurable
  trait in this archive.** A club's venue gap (points-per-game at home minus
  away) has *split-half correlation of −0.03* — its own other half-season is
  not predicted by it at all — and the club-to-club spread (SD 0.373 ppg) is
  *smaller* than pure sampling noise would produce (0.431 ppg expected from 19
  matches/venue). This closes the "away-specific team strength" hypothesis
  that `FINDINGS_MARKET.md` identified as the single best-evidenced remaining
  lead (61.8% of the market gap sits on away wins). — internal,
  `ml/FINDINGS_VENUE.md:51-104`
- **League-level home advantage has genuinely fallen over the 26-season
  archive** (~0.30 → ~0.20 on the log scale) and the served rating's frozen
  online-fitted HFA lags it. Correcting the drift buys **+0.00035, null**
  (interval through zero) — it fixes a real, correctly-diagnosed bias but the
  bias itself is small. Slower-tracking corrections beat faster ones (the
  10×-learning-rate arm beat the 100× arm and a 380-fixture rolling estimate),
  because a single season's HFA estimate carries ~0.03 of noise against a
  ~0.10 drift over two decades. — internal, `ml/FINDINGS_VENUE.md:107-177`
- **The away-win prediction gap remains unexplained** after removing both the
  club-trait and the drift hypotheses — "whatever identifies a strong away
  side to the closing line is not a venue parameter and not a calibration
  constant." Still open going into v4. — internal, `ml/FINDINGS_VENUE.md:205-230`

### A6. Market benchmark and calibration (`ml/FINDINGS_MARKET.md`)

- **The gap to the de-vigged Pinnacle closing line is 0.0211 (rolling)** —
  about 2.8× everything the entire ratings phase bought. The model has closed
  80.4% of the distance from class priors to the market; **19.6 percentage
  points remain**. — internal, `ml/FINDINGS_MARKET.md:26-64`
- **The gap is information, not calibration.** Marginal calibration
  (home/draw/away base rates) is fine. But bucketed by the market's own
  ranking, the model is systematically *under-confident exactly where the
  market is most confident* (market says 0.6674–0.9 in the top decile,
  actual 0.6881, model only 0.5872). **Temperature scaling makes things
  worse** (T=0.945 costs −0.00039 held out) — this rules out any pure
  recalibration-layer fix; the model needs actual new discriminating signal,
  not sharpening. — internal, `ml/FINDINGS_MARKET.md:83-121`
- **61.8% of the market gap is on away wins, 35.5% on draws, only 2.7% on home
  wins** — the model is essentially at parity with the market when the home
  side wins. — internal, `ml/FINDINGS_MARKET.md:126-150`

### A7. Synthesis — where v4 effort is best directed, per the project's own evidence

1. The single largest unexploited, *already-identified* lever is refitting the
   Dixon-Coles rates on a live/production cadence and re-measuring the blend
   weight online rather than freezing it (open items in `FINDINGS_GOALS.md`
   §5, `FINDINGS_CAPACITY.md` item 5) — cheap, already coded, not yet shipped.
2. Away-win prediction is the biggest remaining error concentration (61.8% of
   the market gap) and three separate structural hypotheses for it (club-level
   home advantage, HFA drift, venue-split ratings, coupled venue states) have
   all been tested and rejected. This is the strongest signal that whatever
   fixes away-win prediction is **not** in the rating-update-rule family this
   project has been searching, and external techniques should be judged
   primarily by whether they attack this specifically.
3. Draws (35.5% of the gap) have resisted every structural attempt
   (Dixon-Coles diagonal alone, tighter fit, coupling, bounding) — the one
   thing that helped was blending two independent forecasters, suggesting
   ensemble diversity, not shape refinement, is what's paying off on draws.

---

## Part B — External, not-yet-tried techniques

### B1. Bivariate Poisson with explicit correlation term (Karlis & Ntzoufras)

**Takeaway:** This project has already implemented the Dixon-Coles low-score
correlation *correction* (τ term) and found it real but small (+0.00085
standalone, zero once blended into the served model). The *full* bivariate
Poisson model (a shared third Poisson component inducing correlation across
all scorelines, not just 0-0/1-0/0-1/1-1) is a different, more general
mechanism that has not been tried here.

**Cited findings:**
- Karlis and Ntzoufras (2003) introduced the bivariate Poisson distribution
  for football scores specifically to allow correlation between the two
  teams' goals rather than assuming independence, motivated by improving fit
  and prediction of draws — [Analysis of sports data by using bivariate
  Poisson models](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366)
- Koopman & Lit (2015) extended this to a *dynamic* bivariate Poisson with a
  state-space model, letting attack/defence strengths evolve via additive
  Gaussian uncertainty with fixed autoregressive parameters rather than a
  fixed or slowly-decaying rate — [A Dynamic Bivariate Poisson Model for
  Analysing and Forecasting Match Results in the EPL](https://www.researchgate.net/publication/253236832_A_Dynamic_Bivariate_Poisson_Model_for_Analysing_and_Forecasting_Match_Results_in_the_English_Premier_League)
- The bivariate Poisson approach is described in the football-analytics
  literature as having become a standard benchmark specifically because it
  captures the natural positive correlation between home and away scores
  (both driven partly by shared match-level factors like pace, referee,
  weather) that the Dixon-Coles τ-correction only patches at the low-score
  corner — [MDPI: Predicting Football Match Results Using a Poisson
  Regression Model](https://www.mdpi.com/2076-3417/14/16/7230)

**Inferences:**
- Because this project already found the Dixon-Coles τ correction worth only
  +0.00085 standalone and *zero* once blended, a full bivariate-Poisson shared
  component is likely to land in the same small-and-shape-only regime this
  project has repeatedly found ("the gain is the fit, not the shape"). It is
  the same family of fix (fix the correlation structure of the score matrix)
  that this project's own evidence says is not where the remaining gap lives.
- It could still be worth a cheap standalone probe (a few hours: fit λ3 by
  MLE alongside λ1/λ2) purely because it is a strict generalization of what's
  already shipped and the marginal implementation cost is low — but per this
  project's own placebo discipline, any gain under 0.0002 must be checked
  against the measured seed band before being trusted.

**Gaps:**
- No source found gives a concrete log-loss delta of full bivariate Poisson
  over Dixon-Coles' τ-correction specifically (as opposed to over independent
  Poisson) on a comparable modern top-five-league dataset with de-vigged
  market baseline — most citing sources describe the mechanism, not a
  head-to-head log-loss number against Dixon-Coles.

---

### B2. Dynamic / time-varying Elo and Glicko variants (Glicko-2, Gaussian-process priors)

**Takeaway:** This project rejected static-scalar Elo and goal-difference Elo
decisively in favor of two-parameter attack/defence ratings (+0.0076 vs
+0.0059/+0.0058), and found that all Elo variants are collinear with and
strictly dominated by attack/defence. Dynamic uncertainty-tracking variants
(Glicko-2's rating deviation/volatility, or Gaussian-process time-varying
priors) address a different axis — *rating uncertainty*, not *rating
dimensionality* — that has not been tried in either the Elo or the
attack/defence family here.

**Cited findings:**
- An adaptive Glicko-2 framework for football forecasting improved on the
  standard Glicko-2 baseline by 4.04%, and reported competitive performance
  against market-implied benchmarks with log-loss gaps of only 1.35% in
  multi-class scenarios — [An Adaptive Glicko-2 Rating Framework for
  Probabilistic Football Forecasting and Season Simulation](https://arxiv.org/pdf/2607.01722)
  (Gap: full paper's absolute log-loss numbers and league/season identity
  could not be confirmed by direct fetch — see Gaps.)
- Comparing standard rating systems, one study found Glicko-2 with draw and
  home/league extensions achieving lower log loss on outcome prediction than
  Poisson-regression-based methods (0.5832 vs. 0.5896–0.5949) — note this
  appears to use natural-log-loss-per-class-with-3-outcomes on a different
  scale/base or dataset than this project's reported 0.98-range numbers, so
  the absolute figures are **not directly comparable** without knowing the
  exact loss definition and league set — [search summary, unverified primary
  source not independently confirmed]
- Fahrmeir & Tutz (1994) and Glickman (1999) proposed rating systems with
  time-varying parameters modeled as autoregressive processes, addressing the
  fact that a club's true strength is not constant even within a season —
  [Generalizing the Elo Rating System](https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf)
- A Gaussian Process paired-comparison model was reported to outperform all
  Elo and Glicko variants tested (best log loss 0.631 in that paper's units)
  — [Gaussian Process Priors for Dynamic Paired Comparison Modelling](https://arxiv.org/pdf/1902.07378)
  (Gap: this appears to be for a different sport/dataset — the 0.631 figure
  is far below any football 1X2 log loss reported elsewhere and likely
  reflects a binary win/loss framing or a different scale; treat as directionally
  suggestive only, not as a transferable number.)

**Inferences:**
- Since this project's own attack/defence ratings already strictly dominate
  every Elo variant tested (including the goal-difference-weighted one), a
  straight port of Glicko-2 (a single-scalar-strength system with an added
  uncertainty term) is unlikely to beat `AttackDefence` on strength
  representation alone. The plausible value-add is orthogonal: **explicit
  per-rating uncertainty** that could (a) let the model down-weight or widen
  its own probability for teams with few recent matches (promoted teams,
  teams mid-season-restart) rather than the fixed, convention-set
  `SEASON_REGRESSION`/`PROMOTED_OFFSET` this project currently uses, or (b)
  feed a calibration/variance feature directly to the XGBoost layer.
- Since `FINDINGS_MARKET.md` found that a monotone recalibration
  (temperature scaling) makes things worse — implying the miscalibration is
  a signal problem, not a confidence-scaling problem — a per-fixture
  *uncertainty* feature (which is not a monotone transform of the point
  estimate, but new information about how reliable that estimate is) is a
  structurally different kind of fix than the ones already ruled out, and is
  the most promising untried idea in this bucket.

**Gaps:**
- No source directly measures Glicko-2-style rating-uncertainty as an
  *auxiliary feature* fed into a gradient-boosted classifier already carrying
  attack/defence ratings (which is the shape this project would need); all
  found sources compare rating systems as standalone forecasters.
- Could not verify absolute log-loss magnitude or dataset for the "4.04%
  improvement" and "1.35% gap to market" claims via direct fetch (arxiv PDF
  fetch returned 404); treat those two specific numbers as unverified until
  the paper is re-fetched successfully.

---

### B3. Hierarchical Bayesian team-strength models with shrinkage (Baio & Blangiardo and successors)

**Takeaway:** The mechanism this family offers that this project has not
tried is **principled partial pooling of small-sample teams' attack/defence
parameters toward a league (or divisional) mean, with the pooling amount
learned rather than hand-set** — directly addressing the newcomer/promoted-team
cold-start problem this project currently handles by a convention-set
`PROMOTED_OFFSET`/`SEASON_REGRESSION` rather than a fitted/data-driven
shrinkage amount.

**Cited findings:**
- Baio & Blangiardo (2010) proposed a Bayesian hierarchical Poisson model
  where each team's attack and defence parameters are drawn from
  league-level hyper-distributions, shrinking noisy small-sample estimates
  toward the league mean, tested on Serie A 1991-92 —
  [Bayesian hierarchical model for the prediction of football results](https://discovery.ucl.ac.uk/16040/1/16040.pdf)
- A known defect of the plain hierarchical-shrinkage approach is
  **overshrinkage** — genuinely elite or weak teams get pulled too far toward
  the mean — which the authors and follow-up work addressed with a mixture
  model (allowing teams to belong to different latent "tiers" with their own
  hyper-distribution) that fit Serie A 2007-08 better —
  [Bayesian hierarchical model, mixture extension](https://www.semanticscholar.org/paper/Bayesian-hierarchical-model-for-the-prediction-of-Baio-Blangiardo/1a974c7f4e90d9f56498d2711ebccb9fcc2e09d6)
- Egidi, Pauli & Torelli (2018) later extended this same hierarchical-Poisson
  family with **commensurate priors that blend historical-data-estimated
  scoring rates with bookmaker-odds-implied rates**, allowing the model to
  lean more on the market when historical data is thin and more on history
  when the market is presumably less informed — [Combining historical data
  and bookmakers' odds in modelling football scores](https://arxiv.org/pdf/1802.08848)
  (statistically significant framework; concrete log-loss delta could not be
  extracted from the fetched PDF — see Gaps)
- A follow-up by related authors uses **period-specific commensurate priors**
  to let attack/defence ability evolve with adaptive shrinkage — borrowing
  more strength from history when past and current form align, and adapting
  quickly when a team changes substantially — described in search-summarized
  form as addressing the same time-varying-strength problem Glicko/GP
  approaches target, from a Bayesian-hierarchical angle instead —
  [search summary; primary source not independently confirmed by fetch]

**Inferences:**
- This directly targets a gap this project's own findings flag as real but
  unaddressed: the archive shows 116/211 clubs have gap seasons, promoted
  teams are measurably weaker than incumbents (1.171 vs 1.362 PPG), and
  `PROMOTED_OFFSET`/`SEASON_REGRESSION`/`K` are "set by convention, not
  swept" (`FINDINGS_RATINGS.md` open item 6). A hierarchical/shrinkage
  reformulation would replace those hand-set constants with a fitted
  shrinkage strength — a principled version of exactly the sweep this
  project already flagged as owed but never run.
- Given this project's finding that a fitted MLE beat an online gradient-step
  fit by +0.00323 ("the gain is the fit, not the shape"), the plausible
  analogous claim here is that a *properly fit* shrinkage amount (via
  empirical Bayes or full Bayesian estimation) could similarly outperform the
  current convention-set constants — but the prior evidence suggests the
  win, if any, will again be modest and concentrated in exactly the seasons/
  situations with thin data (new promotions, early-season fixtures), not a
  broad log-loss mover across the whole archive.
- The market-informed-commensurate-prior variant is close in spirit to this
  project's own log-pool blend of the Dixon-Coles score matrix with the tree
  classifier (already the single largest live finding, +0.00184–0.00341
  depending on depth) — it is a different mechanism (blending at the *prior*
  level inside the likelihood, vs. blending at the *output-probability* level
  after two independent fits) worth distinguishing from what's shipped.

**Caveat/risk:** Full Bayesian hierarchical fitting (MCMC/Stan) is
substantially more implementation-heavy than this project's current
closed-form/gradient-update rating replay, and the "replay 26 seasons in
0.4 seconds" serving-latency advantage documented in `FINDINGS_CAPACITY.md`
would likely not survive a full re-fit-per-request Bayesian model — a
production version would need to be fit offline periodically (as the
Dixon-Coles score matrix already is, on a weekly cadence) rather than
replayed online.

**Gaps:**
- Could not extract a concrete log-loss/RPS delta for the commensurate-prior
  bookmaker-blend model from the fetched arXiv PDF (binary/encoding issue in
  fetch); the improvement is described qualitatively in secondary sources but
  no verified number was obtainable in this research pass.
- No source found directly compares hierarchical-Bayesian shrinkage against a
  two-parameter online attack/defence rating of the type already used here,
  on a comparable multi-decade, multi-league dataset.

---

### B4. Low-rank / latent-factor team embeddings

**Takeaway:** This is the least mature and most speculative external
direction found — no source gives a clean, directly comparable log-loss
number for a pure team-level latent-factor model against a Dixon-Coles/
attack-defence baseline on football data; most modern latent-factor work in
this space is player-level (out of scope) or wrapped inside deep-learning
architectures whose gains are hard to attribute to the embedding idea alone.

**Cited findings:**
- Recent work uses team/player embeddings learned from historical match
  statistics combined via transformer architectures, reporting F1-scores
  around 0.47 on Premier League data vs. 0.39 for a bookmaker-odds baseline
  — [PLOS ONE: A framework of interpretable match results prediction in
  football with FIFA ratings and team formation](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0284318)
  (Gap: F1 score is not log loss and not directly convertible; also unclear
  whether the "bookmaker baseline" is de-vigged, which this project's own
  `FINDINGS_MARKET.md` showed materially changes the comparison.)
- Matrix-factorization-style latent factors (as used in recommender systems)
  are described as generally applicable to any two-entity interaction
  problem, but no source found applies this specifically to team-vs-team
  goal-scoring at the fidelity of Dixon-Coles-style rate estimation —
  [Dive into Deep Learning: Matrix Factorization](https://d2l.ai/chapter_recommender-systems/mf.html)
  is a generic reference, not football-specific evidence.

**Inferences:**
- Attack/defence ratings *are*, structurally, already a rank-2 latent-factor
  model per club (one "attack" and one "defence" scalar) fit by an online
  update rule. A genuinely higher-rank latent embedding (rank 4-8, say,
  capturing style axes beyond raw scoring rate — e.g. variance/inconsistency,
  home/away split, tempo) is a real, untried generalization — but this
  project's own evidence that stacking attack/defence with Elo and GD-Elo
  (three different rank-1/rank-2 formulations of "the same underlying
  quantity," r≥0.955) added *nothing* is a strong prior that additional
  latent dimensions derived from the same goals/results signal will be
  redundant with what `AttackDefence` already captures, unless the extra
  dimensions are fit from genuinely new data (e.g., shot maps, passing
  networks — out of scope per player-level exclusion, and this project's own
  RWI investigation already hit a hard data-availability wall trying to get a
  shot-derived signal into the rating state pre-2015).
- The venue-coupling experiment in `FINDINGS_RWI.md` is, in effect, a
  4-dimensional (home-attack, home-defence, away-attack, away-defence) latent
  embedding with a partial-coupling regularizer between the home/away pairs
  — and the entire coupling axis from independent to fully-shared was found
  dead. This is direct, already-run evidence against the most obvious
  "slightly higher rank" extension of the current model.

**Gaps:**
- No football-specific source found reports a controlled ablation isolating
  "rank of the latent factor" as the only variable (holding data and fitting
  method fixed) with log-loss as the metric — this remains genuinely
  unexplored territory in the literature, not just in this project.

---

### B5. Skellam distribution — directly modeling goal difference

**Takeaway:** A well-established alternative to (bivariate) Poisson that
models goal *difference* directly rather than deriving it from two goal
counts; it is a different parameterization of essentially the same
information this project's Dixon-Coles rates already carry, so it is
unlikely to add new signal, but it may be a cheaper or more numerically
convenient way to get 1X2 probabilities and has an explicit handicap-market
interpretability advantage.

**Cited findings:**
- Karlis & Ntzoufras proposed modeling the goal difference directly via the
  Skellam distribution (the distribution of the difference of two Poisson
  variables), validated on EPL 2006-07 data, showing it captures both
  overdispersion and the intrinsic home/away scoring dependence without
  needing an explicit bivariate correlation term — [Bayesian modelling of
  football outcomes: using the Skellam's distribution for the goal
  difference](https://www.academia.edu/1008427/Bayesian_modelling_of_football_outcomes_using_the_Skellams_distribution_for_the_goal_difference)
- The Skellam approach naturally produces win/draw/loss probabilities and
  interpretable parameters useful for Asian-handicap-style markets, but
  cannot reproduce exact scorelines the way a full bivariate score matrix
  can — [search summary of the same paper family]

**Inferences:**
- Since Skellam is mathematically a specific way of marginalizing exactly the
  bivariate-Poisson-style model this project already fits (Dixon-Coles rates
  →score matrix→ read off diagonal for draws), and since this project's own
  finding was that the score-matrix *shape itself earned nothing* over the
  raw rates ("dc_probs against dc_rates: +0.00023, null"), a Skellam
  reparameterization is very unlikely to move log loss on its own. Its
  practical value here, if any, would be implementation simplicity (skip
  building the full score matrix if only 1X2 output is needed) — an
  engineering, not a modeling, win.

**Gaps:**
- No source found gives a Skellam-vs-full-bivariate-Poisson log-loss
  head-to-head on a comparable dataset; the papers found position Skellam as
  a computational/interpretability alternative rather than an accuracy
  improvement.

---

### B6. Shrinkage/regularization for small-sample and promoted/relegated teams (beyond hierarchical Bayes)

**Takeaway:** Outside the formal hierarchical-Bayesian route (B3), the
football-rating-systems literature has a simpler, more implementation-cheap
convention worth testing directly against this project's current
constant-offset approach: seeding a promoted/relegated team's post-transition
rating from the **empirical average rating of teams that made the same
transition in prior seasons**, rather than a single hand-set offset/discount.

**Cited findings:**
- Some rating-system implementations assign a team relegated into a league
  the average rating of teams that were *promoted out of* that league the
  previous season, and vice versa for promoted teams — i.e., a
  transition-conditional empirical prior rather than a fixed points/Elo
  offset — [search summary; specific primary source (Club Elo or a similar
  methodology page) not independently confirmed by direct fetch in this
  pass]
- The direction of adjustment (positive for promotion, negative for
  relegation, tracking that promoted-team budgets generally rise and
  relegated-team budgets generally fall) is consistent with what this
  project independently measured empirically (newcomers score 1.171 PPG vs.
  1.362 for incumbents) — [internal: `ml/FINDINGS_RATINGS.md:176-180`]

**Inferences:**
- This project's `PROMOTED_OFFSET` is already directionally correct and
  empirically validated in sign and rough size by its own diagnostics. The
  marginal value of switching to a transition-conditional empirical average
  (vs. the current single constant) is likely small, because the same
  archive limitation that makes cross-league comparison impossible (zero
  clubs playing in more than one league, five disconnected components) also
  limits how much a "what did *this specific pair of leagues'* promotion/
  relegation transitions look like historically" prior could differ from a
  single well-chosen constant, absent enough transition events per
  league-pair to estimate it reliably.
- Still, this is one of the cheapest untried ideas on the whole list:
  implementation is a lookup table update to `ratings_lib.py`'s
  `season_boundary()`, and this project's own open item 6 in
  `FINDINGS_RATINGS.md` already flags `PROMOTED_OFFSET`/`SEASON_REGRESSION`/
  `K` as unswept.

**Gaps:**
- No direct log-loss evidence found (from this project or externally) for
  transition-conditional vs. fixed-constant promoted-team seeding, isolated
  from other rating-mechanism changes.

---

### B7. Market-informed priors / blending with betting-market information

**Takeaway:** This is the most externally validated, highest-confidence
family in this research pass, and it is philosophically the closest to what
this project already found works best (the Dixon-Coles/XGBoost log-pool
blend) — but this project has so far only used the market for *evaluation*
(the closing-line benchmark in `FINDINGS_MARKET.md`), never as a *model
input*.

**Cited findings:**
- Egidi, Pauli & Torelli (2018) built a hierarchical Bayesian Poisson model
  where each team's scoring rate is a convex combination of a
  historical-data-estimated rate and a bookmaker-odds-implied rate, framed
  explicitly as an underused source of predictive information despite the
  well-known strong relationship between football models and betting markets
  — [Combining historical data and bookmakers' odds in modelling football
  scores](https://www.researchgate.net/publication/328490567_Combining_historical_data_and_bookmakers'_odds_in_modelling_football_scores)
- The general finding that betting markets are highly informative pre-match
  forecasters, and that model parameters can be calibrated toward
  market-implied forecasts, appears repeatedly in this literature —
  including work explicitly calibrating team-strength parameters to
  approximately match market forecasts from Betfair —
  [search summary citing a market-calibration approach; primary source not
  independently confirmed]
- This project's own `FINDINGS_MARKET.md` independently confirms the
  premise: **the model's probabilities are compressed relative to the
  market specifically where the market is most confident** (model 0.5872 vs.
  market 0.6674 vs. actual 0.6881 in the top decile), and this is
  *information the model doesn't have*, not a calibration artifact (temperature
  scaling made it worse). — internal, `ml/FINDINGS_MARKET.md:83-121`

**Inferences:**
- This is the single external idea in this research pass most directly
  supported by this project's own diagnostic work. `FINDINGS_MARKET.md`
  essentially proved the *precondition* for a market-informed prior to help:
  the market has genuine extra signal the model lacks, and no
  transformation of the model's own output can recover it. A commensurate-
  prior or convex-combination approach that feeds **opening-line** odds
  (not closing, to avoid leaking post-lineup information the model
  shouldn't have access to at prediction time, and to preserve a clean
  backtest) into the rate/rating estimation stage is a plausible, well-
  motivated v4 candidate.
- Caveat/risk, drawn from this project's own market-benchmark work: the
  odds-history *sourcing* problem is real and was already solved once —
  `FINDINGS_MARKET.md` found the paid Odds API cost prohibitive/incomplete
  (56.6% coverage, ~$59/month) and switched to free football-data.co.uk CSVs
  for closing lines. **Opening lines**, which is what a market-informed prior
  would need to avoid a look-ahead/leakage risk, may have different
  historical coverage and would need the same due-diligence pass
  `FINDINGS_MARKET.md` already did for closing lines. This is a data-
  availability risk to validate before committing engineering time.
- A second caveat: using the market as an input rather than only as an
  evaluation benchmark changes what "beating the market" even means for this
  project's own promotion bar — if the model partly *is* the market, log
  loss vs. the closing line stops being an independent measure of the
  model's own skill. This is a methodological, not just implementation,
  decision that should be made explicitly before building the arm.

**Gaps:**
- Could not extract the concrete log-loss/RPS improvement number from Egidi
  et al.'s paper via direct PDF fetch in this pass (fetch returned
  unreadable/encoded content); the qualitative claim (odds-blended model
  improves fit and predictive accuracy over historical-data-only) is
  well-attested across multiple citing sources but the effect size is not
  independently verified here.
- No source found directly measures a market-informed prior against a
  Dixon-Coles/XGBoost ensemble of this project's specific type (i.e., where
  the market would be a *third* input alongside two already-blended
  forecasters) — the closest analogues blend market with a single Poisson
  model, not with an existing two-forecaster ensemble.

---

## Cross-cutting caveats for any v4 candidate (drawn from this project's own methodology)

- **Effect-size floor.** `FINDINGS_CAPACITY.md` measured an SD of
  0.00006–0.00007 in pooled log loss from re-seeding alone, with 11% of
  same-recipe seed pairs clearing the project's own 95% bar. Any v4 technique
  promising a gain under ~0.0002 must be checked against a seed-band
  re-sample, not just a single-seed paired bootstrap, or it risks being
  another false positive on the board.
- **Placebo discipline.** Every real-looking column/feature gain in this
  project's history has needed a matched placebo (same shape, shuffled/
  randomized content) before being trusted — twice, a noise placebo cleared
  significance because of implicit regularization in an over-capacity tree,
  not because the feature was informative. Any new architecture (bivariate
  Poisson term, embedding dimension, market-blend weight) should be
  pre-registered with a placebo arm of matching parameter count.
- **State-replay correctness.** Any technique that adds more sequential state
  to the rating update (coupled venue states, hierarchical time-varying
  priors, Glicko-style uncertainty tracking) inherits the train/serve
  equivalence risk already found once in this project (`(date, fixture_id)`
  stable-sort bug moving a single fixture's rating by 2.5 SD, invisible in
  pooled metrics) — needs the same byte-identical-replay test discipline
  used for `ratings_state.py`.
- **Where the market gap actually is.** 61.8% away wins, 35.5% draws, 2.7%
  home wins. Any v4 candidate should be evaluated on the by-outcome
  breakdown, not just pooled log loss — several ideas in this project's own
  history (venue coupling, residual bounding) looked flat or slightly
  positive in pooled terms while actively making the away-win problem worse.

---

## Sources consulted

- Internal: `ml/FINDINGS_CAPACITY.md`, `ml/FINDINGS_RATINGS.md`,
  `ml/FINDINGS_RWI.md`, `ml/FINDINGS_GOALS.md`, `ml/FINDINGS_VENUE.md`,
  `ml/FINDINGS_MARKET.md` (all read in full), plus `git log` history in
  `/home/taiki/Desktop/projects/footyPredict`.
- [Analysis of sports data by using bivariate Poisson models (Karlis & Ntzoufras, 2003)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/1467-9884.00366)
- [A Dynamic Bivariate Poisson Model for the EPL (Koopman & Lit)](https://www.researchgate.net/publication/253236832_A_Dynamic_Bivariate_Poisson_Model_for_Analysing_and_Forecasting_Match_Results_in_the_English_Premier_League)
- [Predicting Football Match Results Using a Poisson Regression Model (MDPI)](https://www.mdpi.com/2076-3417/14/16/7230)
- [An Adaptive Glicko-2 Rating Framework for Probabilistic Football Forecasting and Season Simulation](https://arxiv.org/pdf/2607.01722)
- [Generalizing the Elo Rating System](https://www-users.york.ac.uk/~bp787/Generalizing_Elo_arxiv.pdf)
- [Gaussian Process Priors for Dynamic Paired Comparison Modelling](https://arxiv.org/pdf/1902.07378)
- [Bayesian hierarchical model for the prediction of football results (Baio & Blangiardo, 2010)](https://discovery.ucl.ac.uk/16040/1/16040.pdf)
- [Combining historical data and bookmakers' odds in modelling football scores (Egidi, Pauli & Torelli, 2018)](https://arxiv.org/pdf/1802.08848)
- [Dixon-Coles and xG: together at last (statsandsnakeoil)](https://www.statsandsnakeoil.com/2018/06/22/dixon-coles-and-xg-together-at-last/)
- [Bayesian modelling of football outcomes: using the Skellam's distribution for the goal difference (Karlis & Ntzoufras)](https://www.academia.edu/1008427/Bayesian_modelling_of_football_outcomes_using_the_Skellams_distribution_for_the_goal_difference)
- [A framework of interpretable match results prediction in football with FIFA ratings and team formation (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0284318)
- [Football Predictions: Kaggle competition (Sportmonks)](https://www.sportmonks.com/blogs/football-predictions-kaggle-competition/)
