# Technical Design Document: CageQuant UFC Decision Engine

## 1. System Overview & Component Architecture

CageQuant is a modular decision intelligence framework for 10-second discrete fight simulation, probability modeling, stochastic risk calculation, and verified agentic analytics.

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       SYSTEM COMPONENT FLOW MATRIX                          │
├──────────────────┬──────────────────────────────────────────────────────────┤
│ Layer            │ Core Modules & Responsibilities                          │
├──────────────────┼──────────────────────────────────────────────────────────┤
│ Presentation     │ app.py (Streamlit UI, Plotly animated dual-axis charts) │
│ Agentic AI       │ src/agent/ (tools.py, copilot.py, auditor.py)            │
│ ML Scoring       │ src/engine/models.py (Calibrated XGBoost Classifier)     │
│ Risk & Tradeoffs │ src/engine/simulation.py, stances.py, monte_carlo.py     │
│ Unsupervised ML  │ src/engine/archetypes.py (PCA + K-Means + KNN)           │
│ Data / Governance│ src/engine/etl.py (DuckDB), config/metrics.yml (Semantic) │
└──────────────────┴──────────────────────────────────────────────────────────┘