import time
from typing import Tuple
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

# --- FLOATING ROBOT CHATBOT STYLING ---
st.markdown(
    """
    <style>
    /* Anchor the floating robot popover button in the bottom-right corner */
    div[data-testid="stPopover"] {
        position: fixed;
        bottom: 28px;
        right: 28px;
        z-index: 9999;
    }
    div[data-testid="stPopover"] > button {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        font-size: 30px;
        background: linear-gradient(135deg, #ef4444 0%, #991b1b 100%);
        color: white;
        border: 2px solid #fca5a5;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.45);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stPopover"] > button:hover {
        transform: scale(1.08);
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.65);
    }
    /* Hide the default downward caret icon inside the circular robot button */
    div[data-testid="stPopover"] > button svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing DuckDB Telemetry Engine...")
def get_warehouse(_version: str = "v8_cagebot_cum_stack") -> OctagonDataEngine:
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
    live_playback = st.toggle("Enable Live Accumulation Playback", value=True)
    speed_label = st.select_slider(
        "Playback Speed",
        options=["0.5x (Slow-Mo)", "1x (Tactical)", "2x (Fast)", "5x (Turbo)"],
        value="2x (Fast)",
        disabled=not live_playback,
    )
    speed_map = {
        "0.5x (Slow-Mo)": 0.22,
        "1x (Tactical)": 0.12,
        "2x (Fast)": 0.05,
        "5x (Turbo)": 0.02,
    }
    playback_delay = speed_map[speed_label]

    st.button("▶️ Replay / Resimulate Bout", type="primary", use_container_width=True)

# --- LOAD FIGHTER PROFILES ---
prof_a = warehouse.load_fighter_profile(fighter_a)
prof_b = warehouse.load_fighter_profile(fighter_b)

# --- RUN SIMULATION & MONTE CARLO ---
sim_engine = OctagonSimulationEngine(db_engine=warehouse)
single_bout = sim_engine.run_full_fight(prof_a, prof_b, max_rounds=max_rounds, stochastic=True)

mc_engine = MonteCarloRiskEngine(num_simulations=num_simulations)
mc_summary = mc_engine.run_bout_simulation(prof_a, prof_b, max_rounds=max_rounds)

logs_df = pd.DataFrame([vars(log) for log in single_bout.interval_logs]).head(max_exchanges).copy()

# Compute monotonically ascending cumulative damage output per fighter
if not logs_df.empty:
    if "damage_dealt_a" in logs_df.columns and "damage_dealt_b" in logs_df.columns:
        logs_df["cum_dmg_a"] = logs_df["damage_dealt_a"].clip(lower=0.0).cumsum()
        logs_df["cum_dmg_b"] = logs_df["damage_dealt_b"].clip(lower=0.0).cumsum()
    else:
        step_dmg_a = (
            -logs_df["fighter_b_hp"]
            .diff()
            .fillna(100.0 - logs_df["fighter_b_hp"].iloc[0])
        ).clip(lower=0.0)
        step_dmg_b = (
            -logs_df["fighter_a_hp"]
            .diff()
            .fillna(100.0 - logs_df["fighter_a_hp"].iloc[0])
        ).clip(lower=0.0)
        logs_df["cum_dmg_a"] = step_dmg_a.cumsum()
        logs_df["cum_dmg_b"] = step_dmg_b.cumsum()

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


def build_figures(
    df_slice: pd.DataFrame, total_len: int, max_total_dmg: float
) -> Tuple[go.Figure, go.Figure]:
    """Constructs the live Win Probability Trajectory and 2-Fighter Cumulative Damage Stacked Bars."""
    x_max = max(2, total_len)

    # 1. Live Win Probability Trajectory
    fig_prob = go.Figure()
    fig_prob.add_trace(
        go.Scatter(
            x=df_slice["interval_id"],
            y=df_slice["p_win_a"] * 100.0,
            name=fighter_a,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(width=3, color="#ef4444"),
        )
    )
    fig_prob.add_trace(
        go.Scatter(
            x=df_slice["interval_id"],
            y=(1.0 - df_slice["p_win_a"]) * 100.0,
            name=fighter_b,
            mode="lines+markers",
            marker=dict(size=5),
            line=dict(width=3, color="#3b82f6"),
        )
    )
    fig_prob.update_layout(
        title="1. Live Win Probability Trajectory (%)",
        xaxis_title="10-Sec Combat Exchange",
        yaxis_title="Win Probability (%)",
        xaxis=dict(range=[1, x_max]),
        yaxis=dict(range=[0, 100]),
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 2. Cumulative Sum of Damage Output Differentiated by Fighter (Stacked Bar)
    fig_stack = go.Figure()
    fig_stack.add_trace(
        go.Bar(
            x=df_slice["interval_id"],
            y=df_slice["cum_dmg_a"],
            name=f"{fighter_a} (Red Corner)",
            marker_color="#ef4444",
        )
    )
    fig_stack.add_trace(
        go.Bar(
            x=df_slice["interval_id"],
            y=df_slice["cum_dmg_b"],
            name=f"{fighter_b} (Blue Corner)",
            marker_color="#3b82f6",
        )
    )
    fig_stack.update_layout(
        barmode="stack",
        bargap=0.18,
        title="2. Cumulative Damage Output by Fighter (Stacked)",
        xaxis_title="10-Sec Combat Exchange",
        yaxis_title="Cumulative Damage Dealt",
        xaxis=dict(range=[0.5, x_max + 0.5]),
        yaxis=dict(range=[0, max(100.0, max_total_dmg * 1.1)]),
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig_prob, fig_stack


# --- LIVE PROGRESSIVE RENDERING SLOTS ---
clock_slot = st.empty()
col_left, col_right = st.columns(2)
slot_prob = col_left.empty()
slot_stack = col_right.empty()

if not logs_df.empty:
    total_steps = len(logs_df)
    max_combined_dmg = float((logs_df["cum_dmg_a"] + logs_df["cum_dmg_b"]).max())

    if live_playback:
        step_size = max(1, total_steps // 35)
        frame_indices = list(range(step_size, total_steps, step_size))
        if not frame_indices or frame_indices[-1] != total_steps:
            frame_indices.append(total_steps)

        for frame_idx, end_idx in enumerate(frame_indices):
            df_slice = logs_df.iloc[:end_idx]
            curr_ex = int(df_slice["interval_id"].iloc[-1])
            curr_rd = ((curr_ex - 1) // 30) + 1
            clock_slot.caption(
                f"⏱️ **Live Octagon Clock:** Round {curr_rd} • Exchange {curr_ex} / {total_steps}"
            )
            fig_prob, fig_stack = build_figures(df_slice, total_steps, max_combined_dmg)
            slot_prob.plotly_chart(fig_prob, use_container_width=True, key=f"prob_live_{frame_idx}")
            slot_stack.plotly_chart(fig_stack, use_container_width=True, key=f"stack_live_{frame_idx}")
            time.sleep(playback_delay)
    else:
        curr_ex = int(logs_df["interval_id"].iloc[-1])
        curr_rd = ((curr_ex - 1) // 30) + 1
        clock_slot.caption(
            f"⏱️ **Final Octagon Clock:** Round {curr_rd} • Exchange {curr_ex} / {total_steps}"
        )
        fig_prob, fig_stack = build_figures(logs_df, total_steps, max_combined_dmg)
        slot_prob.plotly_chart(fig_prob, use_container_width=True, key="prob_static")
        slot_stack.plotly_chart(fig_stack, use_container_width=True, key="stack_static")


# ==============================================================================
# FLOATING PERSONAL SIMULATOR ASSISTANT: "CAGEBOT"
# ==============================================================================
with st.popover("🤖", help="Chat with CageBot — Your Personal Simulator Assistant"):
    st.markdown("### 🤖 CageBot • Simulator Assistant")

    with st.chat_message("assistant"):
        st.markdown(
            f"Hey! I'm **CageBot**, your personal assistant for this simulator. "
            f"We're currently looking at **{fighter_a}** vs. **{fighter_b}** ({max_rounds} rounds, "
            f"{num_simulations} Monte Carlo sims). What can I do for you? **AMA!**"
        )

    user_prompt = st.text_area(
        "Your Question:",
        placeholder=f"e.g., Who has the late-round cardio edge between {fighter_a} and {fighter_b}?",
        height=80,
        label_visibility="collapsed",
    )

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        use_mock = st.checkbox("Fast Local Mode ($0)", value=True)
    with col_c2:
        force_hallucination = st.checkbox("Test Auditor Gate", value=False)

    if st.button("💬 Ask CageBot", type="primary", use_container_width=True):
        active_query = user_prompt.strip() or (
            f"Give me a friendly, clear breakdown of win probabilities, finish risks, and pacing "
            f"between {fighter_a} and {fighter_b} across {max_rounds} rounds."
        )

        with st.chat_message("user"):
            st.markdown(active_query)

        copilot = DecisionCopilot(use_mock=use_mock)
        result = copilot.execute_fight_analysis(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            user_query=active_query,
            force_hallucination=force_hallucination,
            num_simulations=num_simulations,
        )

        with st.chat_message("assistant"):
            if result.status == "success":
                st.markdown(
                    f"Here's what the simulation data shows for **{fighter_a} vs. {fighter_b}**:\n\n"
                    f"{result.narrative}"
                )
                st.caption("✅ *Verified against Monte Carlo & DuckDB ground-truth receipt (±1.5% tolerance).*")
            else:
                st.error(f"🛑 Hold on—my Receipt Auditor flagged an issue ({result.status.upper()}):")
                st.markdown(result.narrative)
                if result.violations:
                    for violation in result.violations:
                        st.warning(f"- {violation}")

        if result.ground_truth:
            with st.expander("🧾 View Underlying Simulation Receipt"):
                st.json(result.ground_truth)
