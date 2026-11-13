"""Plain-Python analytical tools (the single source of truth).

`mcp_server.py` only wraps these with FastMCP; `harness.py` calls them directly.
"""

from typing import Any, Dict

from src.agent.schemas import MetricQueryResponse, SimulationResponse
from src.engine.etl import OctagonDataEngine
from src.engine.monte_carlo import MonteCarloRiskEngine


def _seeded_db() -> OctagonDataEngine:
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    return db_engine


def query_semantic_metric(fighter_name: str, metric_name: str) -> Dict[str, Any]:
    """Reads one governed metric for a fighter from DuckDB.

    Args:
        fighter_name: Fighter present in the DuckDB profile table.
        metric_name: Attribute name of the fighter profile (e.g. "punch_power").

    Returns:
        A MetricQueryResponse as a dict.

    Raises:
        ValueError: If the fighter or the metric does not exist. Failing loudly
            matters: silently returning 0.0 would feed the auditor a fake number.
    """
    profile = _seeded_db().load_fighter_profile(fighter_name)
    if not hasattr(profile, metric_name):
        raise ValueError(f"Unknown metric '{metric_name}'.")
    value = round(float(getattr(profile, metric_name)), 4)
    return MetricQueryResponse(fighter_name=fighter_name, metric_name=metric_name, value=value).model_dump()


def run_stochastic_simulation(fighter_a: str, fighter_b: str, num_simulations: int = 1000) -> Dict[str, Any]:
    """Runs N Monte Carlo bouts and returns win-rate distributions.

    Raises:
        ValueError: If either fighter is not in the DuckDB profile table.
    """
    db_engine = _seeded_db()
    prof_a = db_engine.load_fighter_profile(fighter_a)
    prof_b = db_engine.load_fighter_profile(fighter_b)

    summary = MonteCarloRiskEngine(num_simulations=num_simulations).run_bout_simulation(prof_a, prof_b)
    return SimulationResponse(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        total_simulations=num_simulations,
        win_rate_a=summary.win_rate_a,
        win_rate_b=summary.win_rate_b,
        draw_rate=summary.draw_rate,
        avg_finish_interval=summary.avg_finish_interval,
        method_breakdown=summary.method_breakdown,
    ).model_dump()
