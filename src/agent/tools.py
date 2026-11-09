from typing import Dict, Any
from src.engine.etl import OctagonDataEngine
from src.engine.monte_carlo import MonteCarloRiskEngine

def query_semantic_metric(fighter_name: str, metric_name: str) -> Dict[str, Any]:
    """Queries governed metric values from the DuckDB analytical data store."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    profile = db_engine.load_fighter_profile(fighter_name)
    val = getattr(profile, metric_name, 0.0)
    return {
        "fighter_name": fighter_name,
        "metric_name": metric_name,
        "value": round(float(val), 4)
    }

def run_stochastic_simulation(fighter_a: str, fighter_b: str, num_simulations: int = 100) -> Dict[str, Any]:
    """Executes N stochastic Monte Carlo bout iterations to calculate win probability distributions."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    
    prof_a = db_engine.load_fighter_profile(fighter_a)
    prof_b = db_engine.load_fighter_profile(fighter_b)
    
    mc_engine = MonteCarloRiskEngine(num_simulations=num_simulations)
    summary = mc_engine.run_bout_simulation(prof_a, prof_b)
    
    return {
        "fighter_a": fighter_a,
        "fighter_b": fighter_b,
        "win_rate_a": summary.win_rate_a,
        "win_rate_b": summary.win_rate_b,
        "draw_rate": summary.draw_rate,
        "avg_finish_interval": summary.avg_finish_interval,
        "method_breakdown": summary.method_breakdown
    }