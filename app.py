import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.agent.copilot import DecisionCopilot
from src.engine.archetypes import FighterArchetypeEngine
from src.engine.etl import DATA_PROVENANCE, OctagonDataEngine
from src.engine.monte_carlo import MonteCarloRiskEngine
from src.engine.simulation import OctagonSimulationEngine
from src.engine.stances import TacticalStanceEngine

st.set_page_config(
    page_title="MMA Decision Intelligence Platform",
    page_icon="🥋",
    layout="wide",
)


@st.cache_resource
def get_warehouse() -> OctagonDataEngine:
    engine = OctagonDataEngine()
    engine.seed_historical_lightweight_dataset()
    return engine


warehouse = get_warehouse()
roster_df = warehouse.conn.execute(
    "SELECT fighter_name, weight_class, stance FROM gold.dim_fighter ORDER BY fighter_name"
).fetchdf()
FIGHTER_NAMES = roster_df["fighter_name"].tolist()

# --- HEADER ---
st.title("🤖 CageQuant: Agentic UFC Decision Intelligence Engine")
st.caption(
    f"10-Second Discrete Combat Telemetry • Calibrated XGBoost • Monte Carlo Risk Engine • "
    f"Receipt-Verified Agentic Copilot | Data Provenance: `{warehouse.provenance.upper()}`"
)

# --- SIDEBAR CONTROLS & MEDALLION OBSERVABILITY ---
with st.sidebar:
    st.header("⚙️ Bout Configuration")
    default_a = FIGHTER_NAMES.index("Islam Makhachev") if "Islam Makhachev" in FIGHTER_NAMES else 0
    default_b = FIGHTER_NAMES.index("Ilia Topuria") if "Ilia Topuria" in FIGHTER_NAMES else min(1, len(FIGHTER_NAMES) - 1)

    fighter_a = st.selectbox("Fighter A (Red Corner)", FIGHTER_NAMES, index=default_a)
    available_b = [f for f in FIGHTER_NAMES if f != fighter_a]
    idx_b = available_b.index("Ilia Topuria") if "Ilia Topuria" in available_b else 0
    fighter_b = st.selectbox("Fighter B (Blue Corner)", available_b, index=idx_b)

    max_rounds = st.select_slider("Scheduled Rounds", options=[3, 5], value=5)
    num_simulations = st.slider("Monte Carlo Iterations (N)", min_value=100, max_value=2000, value=1000, step=100)

    st.divider()
    st.subheader("🏛️ Medallion Star Schema")
    schema_stats = warehouse.get_star_schema_summary()
    st.caption(f"Provenance Tag: `{schema_stats['provenance']}`")
    counts_df = pd.DataFrame(
        [{"Layer / Table": k, "Rows": v} for k, v in schema_stats["table_counts"].items()]
    )
    st.dataframe(counts_df, hide_index=True, use_container_width=True)

prof_a = warehouse.load_fighter_profile(fighter_a)
prof_b = warehouse.load_fighter_profile(fighter_b)

# --- TABS ---
tab_sim, tab_agent, tab_tactics, tab_warehouse = st.tabs(
    [
        "📊 Telemetry & Monte Carlo Risk",
        "🤖 Agentic Decision Copilot",
        "🧠 Stances & Fighter Archetypes",
        "🗄️ Gold Star Schema Explorer",
    ]
)

# ==============================================================================
# TAB 1: TELEMETRY & MONTE CARLO SIMULATION
# ==============================================================================
with tab_sim:
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀 Run Stochastic Simulation", type="primary", use_container_width=True)
    with col_info:
        st.info(
            f"Simulating **{fighter_a}** vs. **{fighter_b}** across **{num_simulations:,}** stochastic bouts "
            f"({max_rounds} rounds × 30 ten-second intervals)."
        )

    sim_engine = OctagonSimulationEngine(db_engine=warehouse)
    single_bout = sim_engine.run_full_fight(prof_a, prof_b, max_rounds=max_rounds, stochastic=True)
    mc_engine = MonteCarloRiskEngine(num_simulations=num_simulations)
    mc_summary = mc_engine.run_bout_simulation(prof_a, prof_b, max_rounds=max_rounds)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"{fighter_a} Win Prob", f"{mc_summary.win_rate_a * 100:.1f}%")
    m2.metric(f"{fighter_b} Win Prob", f"{mc_summary.win_rate_b * 100:.1f}%")
    m3.metric("Avg Finish Interval", f"{mc_summary.avg_finish_interval:.1f}")
    m4.metric("Sampled Bout Outcome", f"{single_bout.winner} ({single_bout.method})")

    logs_df = pd.DataFrame([vars(log) for log in single_bout.interval_logs])
    if not logs_df.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=logs_df["interval_id"],
                y=logs_df["p_win_a"] * 100.0,
                name=f"{fighter_a} Win Prob (%)",
                mode="lines",
                line=dict(width=3, color="#ef4444"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=logs_df["interval_id"],
                y=logs_df["fighter_a_hp"],
                name=f"{fighter_a} HP",
                mode="lines",
                line=dict(width=2, dash="dot", color="#f97316"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=logs_df["interval_id"],
                y=logs_df["fighter_b_hp"],
                name=f"{fighter_b} HP",
                mode="lines",
                line=dict(width=2, dash="dot", color="#3b82f6"),
            )
        )
        fig.update_layout(
            title=f"10-Second Interval Telemetry Trajectory: {fighter_a} vs. {fighter_b}",
            xaxis_title="10-Second Interval ID",
            yaxis_title="Win Probability (%) / Remaining HP",
            template="plotly_dark",
            height=430,
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2: AGENTIC DECISION COPILOT (PLANNER - EXECUTOR - AUDITOR)
# ==============================================================================
with tab_agent:
    st.subheader("🤖 Planner → Executor → Receipt-Verified Auditor")
    st.markdown(
        "Ask for a tactical telemetry breakdown. Gambling, betting, or deterministic winner-pick prompts "
        "are automatically blocked by the safety guardrail."
    )

    user_prompt = st.text_input(
        "Tactical Query",
        value=f"Analyze the stochastic win distribution and finish risk between {fighter_a} and {fighter_b}.",
    )
    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        use_mock = st.checkbox("Use Deterministic Mock Narrator (Zero API Cost)", value=True)
    with col_a2:
        force_hallucination = st.checkbox("Simulate Hallucination (Test Auditor Gate)", value=False)

    if st.button("🤖 Execute Audited Agent Analysis", type="primary"):
        copilot = DecisionCopilot(use_mock=use_mock)
        result = copilot.execute_fight_analysis(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            user_query=user_prompt,
            force_hallucination=force_hallucination,
            num_simulations=num_simulations,
        )

        if result.status == "success":
            st.success("✅ Auditor Verification Passed — 100% of numeric claims match ground truth.")
            st.write(result.narrative)
        else:
            st.error(f"🛑 Output Refused ({result.status.upper()})")
            st.write(result.narrative)
            if result.violations:
                st.warning("Audit Violations Detected:")
                for v in result.violations:
                    st.write(f"- {v}")

        if result.ground_truth:
            with st.expander("🧾 View Verified MCP Ground-Truth Receipt"):
                st.json(result.ground_truth)

# ==============================================================================
# TAB 3: TACTICAL STANCES & ARCHETYPE CLUSTERING
# ==============================================================================
with tab_tactics:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎯 Tactical Stance Modifiers")
        stance_engine = TacticalStanceEngine()
        stances = list(stance_engine.STANCE_MODIFIERS.keys())
        sel_stance = st.selectbox("Select Tactical Stance", stances)
        mod_prof = stance_engine.apply_stance(prof_a, sel_stance)
        st.json(
            {
                "fighter": mod_prof.name,
                "stance": sel_stance,
                "adjusted_punch_power": round(mod_prof.punch_power, 2),
                "adjusted_arm_block_pct": round(mod_prof.arm_block_pct, 3),
                "adjusted_takedown_def_pct": round(mod_prof.takedown_def_pct, 3),
                "adjusted_stamina_decay": round(mod_prof.stamina_decay_rate, 4),
            }
        )

    with c2:
        st.subheader("🧬 Unsupervised Fighter Archetypes")
        arch_engine = FighterArchetypeEngine()
        all_profiles = [warehouse.load_fighter_profile(name) for name in FIGHTER_NAMES]
        arch_df = arch_engine.classify_roster(all_profiles)
        st.dataframe(arch_df, hide_index=True, use_container_width=True)

# ==============================================================================
# TAB 4: GOLD STAR SCHEMA EXPLORER
# ==============================================================================
with tab_warehouse:
    st.subheader("🗄️ Gold Layer: Star Schema Dimensions & Fact Telemetry")
    st.markdown("**`gold.dim_fighter` (6 Calibrated Roster Profiles)**")
    dim_f = warehouse.conn.execute("SELECT * FROM gold.dim_fighter ORDER BY fighter_sk").fetchdf()
    st.dataframe(dim_f, hide_index=True, use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("**`gold.dim_bout`**")
        st.dataframe(warehouse.conn.execute("SELECT * FROM gold.dim_bout").fetchdf(), hide_index=True, use_container_width=True)
    with col_d2:
        st.markdown("**`gold.fact_interval_telemetry`**")
        st.dataframe(
            warehouse.conn.execute("SELECT * FROM gold.fact_interval_telemetry").fetchdf(),
            hide_index=True,
            use_container_width=True,
        )
