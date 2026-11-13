# 🥊 [Production-Grade UFC Decision Intelligence Platform](https://ufc-mma-cagequant-fightpred-decision-engine.streamlit.app/)

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://modelcontextprotocol.io)
[![Framework](https://img.shields.io/badge/Agentic%20AI-FastMCP-green)](https://modelcontextprotocol.io/)
[![Testing](https://shields.io)](https://pytest.org)
[![Docker](https://shields.io)](https://docker.com)

An enterprise-grade, TDD-backed Decision Intelligence Engine designed for 10-second discrete combat interval telemetry forecasting, stochastic Monte Carlo risk modeling, and receipt-verified Agentic Analytics.

---

## 🏛️ Executive Summary & Engineering Rationale

Traditional sports modeling tools rely on static macro-aggregates (e.g., career strike accuracy or total fight time), ignoring intra-fight momentum shifts, stamina degradation, and tactical adjustments. **CageQuant** solves this by discretizing fight data into 10-second telemetry intervals, enabling dynamic stochastic forecasting and real-time decision intelligence.

### Why This Stack? Architectural Trade-Offs

| Technology Component | Selected Solution | Architecture Rationale & Trade-off |

| :--- | :--- | :--- |

| **Analytical Data Store** | `DuckDB` | Chosen over traditional OLTP (PostgreSQL) and Pandas in-memory arrays. DuckDB provides columnar vectorization for sub-millisecond aggregation across millions of interval records while maintaining zero-server deployment footprints. |

| **Governance Engine** | `metrics.yml` | Implements a headless semantic layer. Separates metric formulas ($xCT$, $SDR$, $SVI$) from database code, guaranteeing that ML models, Streamlit dashboards, and LLM tools query identical business logic. |

| **Predictive ML Core** | `XGBoost` (Calibrated) | Evaluated strictly via **Brier Skill Score** ($BSS > 0.0$) against empirical base rates to eliminate uncalibrated probability estimates in binary classification. |

| **Stochastic Simulator** | Monte Carlo Engine ($N=1,000$) | Injects Gaussian noise ($\mu=1.0, \sigma=0.35$) into interval calculations to capture non-deterministic combat variance (flash knockouts, cut stoppages, referee interventions). |

| **Agentic Framework** | `FastMCP` + Custom `Auditor` | Replaces unconstrained LLM chat with standard Model Context Protocol tool execution. An analytical receipt-auditor validates 100% of generated numerical claims against ground-truth data within a $\pm 1.5\%$ tolerance. |

| **Infrastructure & CI** | Docker + GitHub Actions | Standardizes deployment across Linux environments and enforces strict, zero-cost deterministic TDD gates on every code push. |

---

## 📸 System Architecture Matrix

```text

┌─────────────────────────────────────────────────────────────────────────────┐

│                       SYSTEM COMPONENT FLOW MATRIX                          │

├──────────────────┬──────────────────────────────────────────────────────────┤

│ Layer            │ Core Modules & Responsibilities                          │

├──────────────────┼──────────────────────────────────────────────────────────┤

│ Presentation     │ [app.py](http://app.py) (Streamlit UI, Plotly dual-axis telemetry charts) │

│ Agentic AI       │ src/agent/ ([tools.py](http://tools.py), [copilot.py](http://copilot.py), [auditor.py](http://auditor.py))            │

│ ML Scoring       │ src/engine/[models.py](http://models.py) (Calibrated XGBoost Classifier)     │

│ Risk & Tradeoffs │ src/engine/[simulation.py](http://simulation.py), [stances.py](http://stances.py), monte_[carlo.py](http://carlo.py)     │

│ Unsupervised ML  │ src/engine/[archetypes.py](http://archetypes.py) (PCA + K-Means + KNN)           │

│ Data / Governance│ src/engine/[etl.py](http://etl.py) (DuckDB), config/metrics.yml (Semantic) │

└──────────────────┴──────────────────────────────────────────────────────────┘
