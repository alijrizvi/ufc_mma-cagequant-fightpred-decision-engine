import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.engine.stats import FighterCombatProfile
from src.engine.simulation import OctagonSimulationEngine
from src.engine.etl import OctagonDataEngine
from src.engine.archetypes import FighterArchetypeEngine
from src.engine.stances import TacticalStanceEngine, TacticalStance
from src.engine.monte_carlo import MonteCarloRiskEngine

# Page Setup & Styling
st.set_page_config(page_title="CageQuant | Octagon Decision Engine", page_icon="🥊", layout="wide")

plotly_template = "plotly_dark"

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🥊 CageQuant: Enterprise UFC Decision Engine")
st.caption("10-Second Interval Telemetry | Dynamic Stance Mechanics | Monte Carlo Risk Engine ($N=1,000$)")
st.divider()

# Engine Initialization
db_engine = OctagonDataEngine()
db_engine.seed_historical_lightweight_dataset()

archetype_engine = FighterArchetypeEngine(db_engine)
archetype_engine.fit()

stance_engine = TacticalStanceEngine()
available_fighters = ["Islam Makhachev", "Carlos Prates", "Dustin Poirier", "Justin Gaethje"]

# Sidebar Setup
st.sidebar.header("⚔️ Fighter Setup & Tactical Stances")

with st.sidebar.expander("Fighter A (Red Corner)", expanded=True):
    selected_a = st.selectbox("Select Fighter A", available_fighters, index=0)
    db_profile_a = db_engine.load_fighter_profile(selected_a)
    
    arch_a, _ = archetype_engine.predict_archetype(db_profile_a)
    peers_a = archetype_engine.get_nearest_fighters(db_profile_a, k=2)
    peer_names_a = ", ".join([p["fighter_name"] for p in peers_a])
    
    st.markdown(f"**Archetype:** `{arch_a}`")
    st.caption(f"🤝 **KNN Peer Matches:** {peer_names_a}")
    
    stance_choice_a = st.selectbox(
        "Tactical Stance (A)", 
        options=[s.value for s in TacticalStance],
        index=0,
        key="stance_a"
    )
    
    punch_a = st.slider("Punch Power (A)", 10.0, 40.0, float(db_profile_a.punch_power))
    block_a = st.slider("Arm Block % (A)", 0.0, 0.8, float(db_profile_a.arm_block_pct))
    stamina_decay_a = st.slider("Stamina Decay (A)", 0.01, 0.10, float(db_profile_a.stamina_decay_rate))

with st.sidebar.expander("Fighter B (Blue Corner)", expanded=True):
    selected_b = st.selectbox("Select Fighter B", available_fighters, index=1)
    db_profile_b = db_engine.load_fighter_profile(selected_b)
    
    arch_b, _ = archetype_engine.predict_archetype(db_profile_b)
    peers_b = archetype_engine.get_nearest_fighters(db_profile_b, k=2)
    peer_names_b = ", ".join([p["fighter_name"] for p in peers_b])
    
    st.markdown(f"**Archetype:** `{arch_b}`")
    st.caption(f"🤝 **KNN Peer Matches:** {peer_names_b}")
    
    stance_choice_b = st.selectbox(
        "Tactical Stance (B)", 
        options=[s.value for s in TacticalStance],
        index=0,
        key="stance_b"
    )
    
    punch_b = st.slider("Punch Power (B)", 10.0, 40.0, float(db_profile_b.punch_power))
    block_b = st.slider("Arm Block % (B)", 0.0, 0.8, float(db_profile_b.arm_block_pct))
    stamina_decay_b = st.slider("Stamina Decay (B)", 0.01, 0.10, float(db_profile_b.stamina_decay_rate))

st.sidebar.subheader("⚙️ Simulation Settings")
max_rounds = st.sidebar.slider("Max Simulation Rounds", 1, 5, 5)
num_mc_sims = st.sidebar.slider("Monte Carlo Iterations (N)", 50, 1000, 200, step=50)
animate_sim = st.sidebar.checkbox("⚡ Enable Live Trajectory Playback", value=True)
animation_speed = st.sidebar.select_slider(
    "Playback Pace", 
    options=["Fast (0.10s)", "Medium (0.20s)", "Dramatic (0.30s)"], 
    value="Medium (0.20s)"
)

run_sim_button = st.sidebar.button("🚀 Execute Fight Engine", type="primary")

if run_sim_button:
    raw_profile_a = FighterCombatProfile(
        name=selected_a, punch_power=punch_a, arm_block_pct=block_a, stamina_decay_rate=stamina_decay_a
    )
    raw_profile_b = FighterCombatProfile(
        name=selected_b, punch_power=punch_b, arm_block_pct=block_b, stamina_decay_rate=stamina_decay_b
    )

    enum_stance_a = next(s for s in TacticalStance if s.value == stance_choice_a)
    enum_stance_b = next(s for s in TacticalStance if s.value == stance_choice_b)
    
    profile_a = stance_engine.apply_stance_modifiers(raw_profile_a, enum_stance_a)
    profile_b = stance_engine.apply_stance_modifiers(raw_profile_b, enum_stance_b)

    sim_engine = OctagonSimulationEngine(db_engine=db_engine)
    single_result = sim_engine.run_full_fight(profile_a, profile_b, max_rounds=max_rounds, stochastic=True)

    mc_engine = MonteCarloRiskEngine(num_simulations=num_mc_sims)
    mc_summary = mc_engine.run_bout_simulation(profile_a, profile_b, max_rounds=max_rounds)

    col1, col2, col3, col4 = st.columns([1.2, 2.2, 1.4, 1.2])
    with col1: st.metric("Single Sim Winner", single_result.winner)
    with col2: st.metric("Method of Victory", single_result.method_of_victory)
    with col3: st.metric("MC Expected Win % (A)", f"{mc_summary.win_rate_a * 100:.1f}%")
    with col4: st.metric("MC Expected Win % (B)", f"{mc_summary.win_rate_b * 100:.1f}%")

    st.divider()

    st.subheader(f"🎲 Monte Carlo Risk Distribution (N = {num_mc_sims} Iterations)")
    mc_col1, mc_col2 = st.columns([1.5, 1.0])
    
    with mc_col1:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Bar(
            y=["Win Probability"], x=[mc_summary.win_rate_a * 100], 
            name=f"{selected_a}", orientation="h", marker=dict(color="crimson")
        ))
        fig_mc.add_trace(go.Bar(
            y=["Win Probability"], x=[mc_summary.draw_rate * 100], 
            name="Draw", orientation="h", marker=dict(color="gray")
        ))
        fig_mc.add_trace(go.Bar(
            y=["Win Probability"], x=[mc_summary.win_rate_b * 100], 
            name=f"{selected_b}", orientation="h", marker=dict(color="royalblue")
        ))
        
        fig_mc.update_layout(
            barmode="stack", height=180, template=plotly_template, 
            xaxis=dict(title="Probability %", range=[0, 100]), 
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_mc, width="stretch", key="mc_win_bar")

    with mc_col2:
        st.markdown("**Method of Victory Distribution**")
        df_methods = pd.DataFrame([
            {"Method": k, "Frequency (%)": f"{v * 100:.1f}%"} 
            for k, v in mc_summary.method_breakdown.items()
        ])
        st.dataframe(df_methods, hide_index=True, width="stretch")
        st.caption(f"**95% Duration CI:** Interval {mc_summary.interval_ci_lower} to {mc_summary.interval_ci_upper} (Avg: {mc_summary.avg_finish_interval})")

    st.divider()

    logs = [
        {
            "interval_id": log.interval_id,
            "round": log.round_num,
            "time_sec": log.time_seconds,
            "fighter_a_hp": log.fighter_a_hp,
            "fighter_b_hp": log.fighter_b_hp,
            "p_win_a": log.p_win_a * 100.0,
            "p_win_b": log.p_win_b * 100.0,
            "action_type": log.action_type,
            "damage_a": log.damage_dealt_a,
            "damage_b": log.damage_dealt_b
        }
        for log in single_result.interval_logs
    ]
    df_logs = pd.DataFrame(logs)
    df_logs["cum_damage_a"] = df_logs["damage_a"].cumsum()
    df_logs["cum_damage_b"] = df_logs["damage_b"].cumsum()

    st.subheader("📈 Live Fight Trajectory & XGBoost Win Probability")
    chart_placeholder = st.empty()

    def build_primary_figure(data_slice):
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
            subplot_titles=("Vitality Degradation (HP)", "Rolling XGBoost Win Probability (%)")
        )
        fig.add_trace(go.Scatter(x=data_slice["interval_id"], y=data_slice["fighter_a_hp"], mode="lines", name=f"{selected_a} HP", line=dict(color="red", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_slice["interval_id"], y=data_slice["fighter_b_hp"], mode="lines", name=f"{selected_b} HP", line=dict(color="blue", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data_slice["interval_id"], y=data_slice["p_win_a"], mode="lines", name=f"{selected_a} Win %", line=dict(color="crimson", width=2, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=data_slice["interval_id"], y=data_slice["p_win_b"], mode="lines", name=f"{selected_b} Win %", line=dict(color="royalblue", width=2, dash="dash")), row=2, col=1)
        
        fig.update_layout(height=500, template=plotly_template, showlegend=True, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_xaxes(title_text="10-Second Interval Segment", row=2, col=1)
        fig.update_yaxes(title_text="HP", range=[0, 105], row=1, col=1)
        fig.update_yaxes(title_text="Probability %", range=[0, 100], row=2, col=1)
        return fig

    speed_map = {"Fast (0.10s)": 0.10, "Medium (0.20s)": 0.20, "Dramatic (0.30s)": 0.30}
    delay = speed_map.get(animation_speed, 0.20)

    if animate_sim:
        step_size = max(1, len(df_logs) // 30)
        for i in range(1, len(df_logs) + 1, step_size):
            slice_df = df_logs.iloc[:i]
            chart_placeholder.plotly_chart(build_primary_figure(slice_df), width="stretch", key=f"plotly_anim_frame_{i}")
            time.sleep(delay)

    chart_placeholder.plotly_chart(build_primary_figure(df_logs), width="stretch", key="plotly_final_frame")

    st.subheader("📊 Cumulative Damage Output Over Time")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=df_logs["interval_id"], y=df_logs["cum_damage_a"], mode="lines", 
        name=f"{selected_a} Cum. Damage", fill="tozeroy", 
        line=dict(color="rgba(255, 75, 75, 0.9)", width=2), fillcolor="rgba(255, 75, 75, 0.15)"
    ))
    fig_cum.add_trace(go.Scatter(
        x=df_logs["interval_id"], y=df_logs["cum_damage_b"], mode="lines", 
        name=f"{selected_b} Cum. Damage", fill="tozeroy", 
        line=dict(color="rgba(30, 144, 255, 0.9)", width=2), fillcolor="rgba(30, 144, 255, 0.15)"
    ))

    fig_cum.update_layout(
        height=350, template=plotly_template, 
        xaxis_title="10-Second Interval Segment", yaxis_title="Total Cumulative Damage Points",
        hovermode="x unified", margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_cum, width="stretch", key="plotly_cum_damage")

    with st.expander("🔍 Inspect Full 10-Second Segment Log & Probabilities", expanded=False):
        st.dataframe(df_logs, width="stretch")
else:
    st.info("Select fighters from DuckDB roster, choose tactical stances, and click **Execute Fight Engine**.")