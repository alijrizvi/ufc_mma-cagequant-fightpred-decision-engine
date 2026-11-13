import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.copilot import DecisionCopilot
from src.engine.etl import DATA_PROVENANCE, OctagonDataEngine
from src.engine.monte_carlo import MonteCarloRiskEngine
from src.engine.simulation import OctagonSimulationEngine

st.set_page_config(
    page_title="CageQuant | UFC Decision Intelligence Engine",
    page_icon="🥋",
    layout="wide",
)


@st.cache_resource(show_spinner="Initializing DuckDB Telemetry Engine...")
def get_warehouse(_version: str = "v6_animated_keys") -> OctagonDataEngine:
    """Initializes and caches the DuckDB engine with seeded Lightweight roster data."""
    engine = OctagonDataEngine()
    engine.seed_historical_lightweight_dataset()

    conn = getattr(engine, "conn", getattr(engine, "con", None))
    if conn is not None:
        conn.execute("CREATE SCHEMA IF NOT EXISTS gold;")
        conn.execute("""
            CREATE OR REPLACE VIEW gold.dim_fighter AS
            SELECT
                ROW_NUMBER() OVER (ORDER BY fighter_name) AS fighter_sk,
                fighter_name,
                'Lightweight' AS weight_class,
                CASE
                    WHEN takedown_success_pct >= 0.50 THEN 'Sambo / Grappling'
                    WHEN leg_kick_power >= 18.0 THEN 'Muay Thai / Striker'
                    ELSE 'Orthodox Boxer-Puncher'
                END AS stance
            FROM fighter_profiles;
        """)
    return engine


warehouse = get_warehouse()
roster_df = warehouse.conn.execute(
    "SELECT fighter_name, weight_class, stance FROM gold.dim_fighter ORDER BY fighter_name"
).fetchdf()
FIGHTER_NAMES = roster_df["fighter_name"].tolist()
ARCHETYPE_MAP = dict(zip(roster_df["fighter_name"], roster_df["stance"]))

# --- HEADER ---
provenance_label = getattr(warehouse, "provenance", DATA_PROVENANCE).upper()
st.title("🥋 CageQuant: UFC Octagon Decision Intelligence Engine")
st.caption(
    f"10-Second Discrete Combat Telemetry • Calibrated XGBoost • Monte Carlo Risk Engine • "
    f"Receipt-Verified Agentic Copilot | Data Provenance: `{provenance_label}`"
)

# --- SIDEBAR: BOUT & ANIMATED PLAYBACK CONTROLS ---
with st.sidebar:
    st.header("⚙️ Bout & Simulation Controls")

    default_a = FIGHTER_NAMES.index("Islam Makhachev") if "Islam Makhachev" in FIGHTER_NAMES else 0
    fighter_a = st.selectbox("Fighter A (Red Corner)", FIGHTER_NAMES, index=default_a)
    st.caption(f"Archetype: **{ARCHETYPE_MAP.get(fighter_a, 'MMA')}**")

    available_b = [f for f in FIGHTER_NAMES if f != fighter_a]
    default_b = available_b.index("Ilia Topuria") if "Ilia Topuria" in available_b else 0
    fighter_b = st.selectbox("Fighter B (Blue Corner)", available_b, index=default_b)
    st.caption(f"Archetype: **{ARCHETYPE_MAP.get(fighter_b, 'MMA')}**")

    st.divider()

    max_rounds = st.slider("Scheduled Rounds", min_value=1, max_value=5, value=5, step=1)
    num_simulations = st.slider(
        "Monte Carlo Simulations (N)", min_value=50, max_value=500, value=250, step=50
    )
    max_exchanges = st.slider(
        "Max Combat Exchanges (10s Steps)",
        min_value=30,
        max_value=150,
        value=max_rounds * 30,
        step=30,
    )

    st.divider()
    st.subheader("🎬 Animated Fight Playback")
    live_playback = st.toggle("Enable Animated Playback", value=True)
    speed_label = st.select_slider(
        "Playback Speed",
        options=["0.5x (Slow-Mo)", "1x (Tactical)", "2x (Fast)", "5x (Turbo)"],
        value="2x (Fast)",
        disabled=not live_playback,
    )
    speed_map = {
        "0.5x (Slow-Mo)": 0.20,
        "1x (Tactical)": 0.10,
        "2x (Fast)": 0.04,
        "5x (Turbo)": 0.015,
    }
    playback_delay = speed_map[speed_label]

    replay_btn = st.button("▶️ Replay / Resimulate Bout", type="primary", use_container_width=True)

# --- LOAD FIGHTER PROFILES ---
prof_a = warehouse.load_fighter_profile(fighter_a)
prof_b = warehouse.load_fighter_profile(fighter_b)

# --- RUN SIMULATION & MONTE CARLO ---
sim_engine = OctagonSimulationEngine(db_engine=warehouse)
single_bout = sim_engine.run_full_fight(prof_a, prof_b, max_rounds=max_rounds, stochastic=True)

mc_engine = MonteCarloRiskEngine(num_simulations=num_simulations)
mc_summary = mc_engine.run_bout_simulation(prof_a, prof_b, max_rounds=max_rounds)

logs_df = pd.DataFrame([vars(log) for log in single_bout.interval_logs]).head(max_exchanges)

# --- TOP PROBABILITY BREAKDOWN METRICS ---
st.subheader("📊 Stochastic Win & Method Probability Breakdown")

win_rate_a = mc_summary.win_rate_a * 100.0
win_rate_b = mc_summary.win_rate_b * 100.0

method_counts = getattr(mc_summary, "method_breakdown", None)
if isinstance(method_counts, dict) and method_counts:
    ko_pct = method_counts.get("KO/TKO", 0.0) * 100.0
    sub_pct = method_counts.get("Submission", 0.0) * 100.0
    dec_pct = method_counts.get("Decision", 0.0) * 100.0
else:
    finish_ratio = min(0.85, max(0.20, 1.0 - (mc_summary.avg_finish_interval / (max_rounds * 30.0))))
    ko_pct = finish_ratio * 0.65 * 100.0
    sub_pct = finish_ratio * 0.35 * 100.0
    dec_pct = (1.0 - finish_ratio) * 100.0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"{fighter_a} Win Prob", f"{win_rate_a:.1f}%")
m2.metric(f"{fighter_b} Win Prob", f"{win_rate_b:.1f}%")
m3.metric("KO / TKO Finish Prob", f"{ko_pct:.1f}%")
m4.metric("Submission Prob", f"{sub_pct:.1f}%")
m5.metric("Decision / Avg End", f"{dec_pct:.1f}% • Exchange {mc_summary.avg_finish_interval:.0f}")

bout_winner = getattr(single_bout, "winner", "Draw")
bout_method = (
    getattr(single_bout, "method", None)
    or getattr(single_bout, "win_method", None)
    or getattr(single_bout, "finish_method", None)
    or getattr(single_bout, "outcome", "Decision / Finish")
)

st.caption(
    f"**Sampled Trajectory Outcome:** `{bout_winner}` via `{bout_method}` "
    f"(Simulated across {num_simulations} Monte Carlo iterations)"
)


def render_charts(df_slice: pd.DataFrame, step_key: str) -> None:
    """Renders the 3 primary telemetry and Monte Carlo diagnostic charts with unique keys."""
    current_exchange = int(df_slice["interval_id"].iloc[-1])
    current_round = ((current_exchange - 1) // 30) + 1
    st.caption(f"⏱️ **Octagon Clock:** Round {current_round} • Exchange {current_exchange} / {len(logs_df)}")

    c1, c2, c3 = st.columns(3)

    # Chart 1: Live Win Probability Trajectory
    with c1:
        fig_prob = go.Figure()
        fig_prob.add_trace(
            go.Scatter(
                x=df_slice["interval_id"],
                y=df_slice["p_win_a"] * 100.0,
                name=fighter_a,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=3, color="#ef4444"),
            )
        )
        fig_prob.add_trace(
            go.Scatter(
                x=df_slice["interval_id"],
                y=(1.0 - df_slice["p_win_a"]) * 100.0,
                name=fighter_b,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=3, color="#3b82f6"),
            )
        )
        fig_prob.update_layout(
            title="1. Live Win Probability Trajectory (%)",
            xaxis_title="10-Sec Combat Exchange",
            yaxis_title="Win Prob (%)",
            xaxis=dict(range=[1, max(2, len(logs_df))]),
            yaxis=dict(range=[0, 100]),
            template="plotly_dark",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_prob, use_container_width=True, key=f"prob_chart_{step_key}")

    # Chart 2: HP & Stamina Attrition Curves
    with c2:
        fig_attr = go.Figure()
        fig_attr.add_trace(
            go.Scatter(
                x=df_slice["interval_id"],
                y=df_slice["fighter_a_hp"],
                name=f"{fighter_a} HP",
                mode="lines",
                line=dict(width=2.5, color="#ef4444"),
            )
        )
        fig_attr.add_trace(
            go.Scatter(
                x=df_slice["interval_id"],
                y=df_slice["fighter_b_hp"],
                name=f"{fighter_b} HP",
                mode="lines",
                line=dict(width=2.5, color="#3b82f6"),
            )
        )
        if "fighter_a_stamina" in df_slice.columns:
            fig_attr.add_trace(
                go.Scatter(
                    x=df_slice["interval_id"],
                    y=df_slice["fighter_a_stamina"],
                    name=f"{fighter_a} Stam",
                    mode="lines",
                    line=dict(width=1.5, dash="dot", color="#fca5a5"),
                )
            )
        if "fighter_b_stamina" in df_slice.columns:
            fig_attr.add_trace(
                go.Scatter(
                    x=df_slice["interval_id"],
                    y=df_slice["fighter_b_stamina"],
                    name=f"{fighter_b} Stam",
                    mode="lines",
                    line=dict(width=1.5, dash="dot", color="#93c5fd"),
                )
            )
        fig_attr.update_layout(
            title="2. Health & Stamina Attrition",
            xaxis_title="10-Sec Combat Exchange",
            yaxis_title="Remaining Pool",
            xaxis=dict(range=[1, max(2, len(logs_df))]),
            yaxis=dict(range=[0, 105]),
            template="plotly_dark",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_attr, use_container_width=True, key=f"attr_chart_{step_key}")

    # Chart 3: Monte Carlo Outcome & Finish Method Distribution
    with c3:
        categories = [f"{fighter_a} Win", f"{fighter_b} Win", "KO/TKO", "Submission", "Decision"]
        values = [win_rate_a, win_rate_b, ko_pct, sub_pct, dec_pct]
        colors = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"]

        fig_mc = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.1f}%" for v in values],
                    textposition="auto",
                )
            ]
        )
        fig_mc.update_layout(
            title=f"3. Monte Carlo Distribution (N={num_simulations})",
            yaxis_title="Probability (%)",
            yaxis=dict(range=[0, 105]),
            template="plotly_dark",
            height=360,
            margin=dict(l=20, r=20, t=45, b=20),
        )
        st.plotly_chart(fig_mc, use_container_width=True, key=f"mc_chart_{step_key}")


# --- RENDER 3 CHARTS (ANIMATED PLAYBACK OR STATIC) ---
chart_placeholder = st.empty()

if not logs_df.empty:
    if live_playback:
        step_size = max(1, len(logs_df) // 30)
        for frame_num, end_idx in enumerate(range(step_size, len(logs_df) + step_size, step_size)):
            with chart_placeholder.container():
                render_charts(logs_df.iloc[:end_idx], step_key=f"frame_{frame_num}")
            time.sleep(playback_delay)
    else:
        with chart_placeholder.container():
            render_charts(logs_df, step_key="static_final")

st.divider()

# ==============================================================================
# INTEGRATED AGENTIC DECISION COPILOT (SINGLE-PAGE INTERACTIVE WORKSPACE)
# ==============================================================================
st.subheader("🤖 Agentic Decision Copilot (Receipt-Verified FastMCP)")
st.caption(
    "Ask tactical or risk-distribution questions about the active matchup. Every numeric claim "
    "is cross-checked against the Monte Carlo & DuckDB ground-truth receipt (±1.5% tolerance)."
)

col_q, col_opts = st.columns([3, 1])
with col_q:
    user_prompt = st.text_input(
        "Ask the Copilot",
        value=(
            f"Compare the finish risk and stamina efficiency between {fighter_a} "
            f"and {fighter_b} over {max_rounds} rounds."
        ),
    )
with col_opts:
    use_mock = st.checkbox("Deterministic Mock Mode", value=True)
    force_hallucination = st.checkbox("Test Hallucination Gate", value=False)

if st.button("🧠 Run Audited Copilot Analysis", type="primary"):
    copilot = DecisionCopilot(use_mock=use_mock)
    result = copilot.execute_fight_analysis(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        user_query=user_prompt,
        force_hallucination=force_hallucination,
        num_simulations=num_simulations,
    )

    if result.status == "success":
        st.success("✅ Auditor Verification Passed — All statistical claims match simulation ground truth.")
        st.markdown(result.narrative)
    else:
        st.error(f"🛑 Output Blocked by Receipt Auditor ({result.status.upper()})")
        st.markdown(result.narrative)
        if result.violations:
            st.warning("Audit Violations Detected:")
            for violation in result.violations:
                st.write(f"- {violation}")

    if result.ground_truth:
        with st.expander("🧾 Inspect Verified MCP Ground-Truth Receipt"):
            st.json(result.ground_truth)
