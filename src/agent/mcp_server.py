"""
FastMCP Server exposing DuckDB metric contracts and Monte Carlo simulations.
"""
from fastmcp import FastMCP
from src.agent.schemas import MetricQueryResponse, SimulationResponse
from src.engine.etl import OctagonDataEngine
from src.engine.monte_carlo import MonteCarloRiskEngine

mcp = FastMCP("CageQuant Decision Server")

@mcp.tool()
def query_semantic_metric(fighter_name: str, metric_name: str) -> dict:
    """
    Query headless semantic metrics from DuckDB governed by metrics.yml.
    """
    db = OctagonDataEngine()
    db.seed_historical_lightweight_dataset()
    profile = db.load_fighter_profile(fighter_name)
    val = float(getattr(profile, metric_name, 0.0))
    
    response = MetricQueryResponse(
        fighter_name=fighter_name,
        metric_name=metric_name,
        value=round(val, 4)
    )
    return response.model_dump()

@mcp.tool()
def run_stochastic_simulation(fighter_a: str, fighter_b: str, num_simulations: int = 1000) -> dict:
    """
    Execute N=1,000 Monte Carlo bout iterations under Gaussian damage volatility.
    """
    db = OctagonDataEngine()
    db.seed_historical_lightweight_dataset()
    
    prof_a = db.load_fighter_profile(fighter_a)
    prof_b = db.load_fighter_profile(fighter_b)
    
    mc = MonteCarloRiskEngine(num_simulations=num_simulations)
    summary = mc.run_bout_simulation(prof_a, prof_b)
    
    response = SimulationResponse(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        total_simulations=num_simulations,
        win_rate_a=summary.win_rate_a,
        win_rate_b=summary.win_rate_b,
        draw_rate=summary.draw_rate,
        avg_finish_interval=summary.avg_finish_interval,
        method_breakdown=summary.method_breakdown
    )
    return response.model_dump()

if __name__ == "__main__":
    mcp.run()