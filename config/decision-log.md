# CageQuant Architecture Decision Records (ADRs)

## ADR 001: 10-Second Discrete Interval Simulation Model
* **Date:** 2026-11-01
* **Status:** Accepted
* **Context:** Continuous differential equation fight modeling lacks granular event triggers and real-time probability update hooks required for fight telemetry.
* **Decision:** Segment bout simulation into discrete 10-second intervals (30 intervals per 5-minute round).
* **Consequences:** Enables deterministic state transitions, step-by-step feature logging, and high-frequency win probability updating.

## ADR 002: Four Combat Pillars (xCT, SDR, SVI, Vitality HP)
* **Date:** 2026-11-01
* **Status:** Accepted
* **Context:** Raw strike totals fail to account for power variance, defensive mitigation, and cardio depletion.
* **Decision:** Formulate four normalized domain metrics: Expected Cumulative Damage (xCT), Stamina Drain Ratio (SDR), Strike Volatility Index (SVI), and Vitality (HP).
* **Consequences:** Standardizes feature inputs for predictive ML algorithms and tactical stance modifiers.

## ADR 003: DuckDB Embedded Analytical Engine
* **Date:** 2026-11-02
* **Status:** Accepted
* **Context:** Need high-performance SQL query execution on structured fighter telemetry without deploying external database servers.
* **Decision:** Embedded DuckDB as the in-memory semantic analytical storage layer.
* **Consequences:** Sub-millisecond SQL query execution, zero infrastructure deployment cost, and seamless integration with Pandas DataFrames.

### ADR 004: Test-Driven Verification & Test Coverage Standards
- **Date:** 2026-08-15
- **Status:** Accepted
- **Decision:** Enforce 100% test coverage across core mathematical paths (DuckDB ETL metrics, XGBoost probability calibration, and Receipt Auditor verification) using `pytest --cov=src/engine --cov=src/agent`. Mocked LLM fixtures are used in CI to ensure zero-cost deterministic execution gates.

## ADR 005: Streamlit & Dual-Axis Plotly Trajectory Visuals
* **Date:** 2026-11-03
* **Status:** Accepted
* **Context:** Stakeholders require real-time visual tracking of vitality degradation alongside rolling probability win metrics.
* **Decision:** Build interactive front-end using Streamlit and sub-second Plotly stacked subplots with step-by-step playback animation.
* **Consequences:** High-impact visual storytelling for portfolio reviews with custom playback pacing controls.

## ADR 006: XGBoost Classifier Calibration via Brier Score Loss
* **Date:** 2026-11-04
* **Status:** Accepted
* **Context:** Heuristic prior logits lacked empirical validation on interval telemetry features.
* **Decision:** Train `xgboost.XGBClassifier` on DuckDB telemetry feature matrix $X$ and calibrate output probabilities using Brier Score Loss ($BS \le 0.25$).
* **Consequences:** Provides empirically calibrated interval win probabilities ($P_{Win}$) bound to telemetry states.

## ADR 007: Stochastic Damage Volatility & Monte Carlo Risk Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Deterministic damage calculations caused Monte Carlo $N=1,000$ iterations to collapse into identical single-winner outcomes.
* **Decision:** Inject Gaussian stochastic variance ($\mu=1.0, \sigma=0.35$) into interval strike damage calculations.
* **Consequences:** Monte Carlo runs branch into realistic, smooth probability distributions, confidence intervals, and varied method-of-victory breakdowns.

## ADR 008: Dynamic Tactical Stance Trade-Off Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Static fighter profiles do not reflect dynamic tactical adjustments or physical power-scaling tradeoffs during fight progression.
* **Decision:** Implement `TacticalStanceEngine` modeling combat stances (Standard, High-Output Power, High-Efficiency Flow, Ascended Peak Output) with trade-offs between damage output, stamina decay, and defense.
* **Consequences:** Enables scenario modeling where aggressive power boosts incur cardio penalties unless neutralized by mastered baseline efficiency.

## ADR 009: Unsupervised PCA, K-Means Archetypes, and KNN Peer Engine
* **Date:** 2026-11-05
* **Status:** Accepted
* **Context:** Fighter profiles require automated categorization into intuitive combat archetypes and historical peer matching.
* **Decision:** Implement `FighterArchetypeEngine` using StandardScaler, PCA dimensional reduction, K-Means clustering ($K=4$), and KNN Euclidean distance peer search.
* **Consequences:** Automatically assigns fighters to tactical archetypes and identifies top-2 nearest historical match-ups for any parameter configuration.

## ADR 010: Transition to FastMCP Tool-Calling Agentic Architecture
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Passive text summarization fails to provide verifiable, dynamic tool interaction.
* **Decision:** Implement a modular `src/agent/` package containing FastMCP tool interfaces (`tools.py`), an orchestration copilot (`copilot.py`), and a verification step (`auditor.py`).
* **Consequences:** Enables dynamic tool invocation and multi-step reasoning capabilities.

## ADR 011: Structured Receipt Auditor for Anti-Hallucination Verification
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Regex-only text parsing is brittle against diverse LLM formatting outputs.
* **Decision:** Build a structured receipt validation auditor in `src/agent/auditor.py` that extracts numerical assertions and matches them against tool output states within a $\pm 1.5\%$ tolerance.
* **Consequences:** Prevents hallucinated metrics from rendering in output briefs.

## ADR 012: Deterministic CI Evals with Mocked LLM Interfaces
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Live LLM API calls in CI pipelines introduce latency, financial cost, and non-deterministic test failures.
* **Decision:** Decouple CI test suites using mocked LLM response generators for GitHub Actions, running live API benchmarking in a separate evaluation pipeline (`evals/`).
* **Consequences:** Ensures 100% reproducible, zero-cost CI builds.

## ADR 013: Model Calibration Evaluation via Brier Skill Score (BSS)
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Absolute Brier Score thresholds ($BS \le 0.25$) are easily satisfied by naive $50/50$ random guessing.
* **Decision:** Evaluate model calibration using Brier Skill Score ($BSS = 1 - BS_{\text{model}} / BS_{\text{baseline}}$), asserting $BSS > 0.0$ against empirical baseline win rates.
* **Consequences:** Ensures predictive models provide verifiable statistical lift over random guessing.

## ADR 014: Light-Touch Agent Architecture over Framework Dependency
* **Date:** 2026-11-07
* **Status:** Accepted
* **Context:** Heavy multi-agent frameworks (LangGraph, CrewAI) introduce significant abstractions and dependency overhead.
* **Decision:** Build a lightweight, explicit agent orchestration loop in Python using FastMCP and native SDK calls.
* **Consequences:** Minimizes latency, keeps dependencies lean, and retains complete control over context and prompt assembly.