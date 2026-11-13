"""FastMCP server: thin wrappers over src/agent/tools.py with JSON error payloads."""

from typing import Any, Dict

from fastmcp import FastMCP

from src.agent import tools

mcp = FastMCP("CageQuant Decision Server")


@mcp.tool()
def query_semantic_metric(fighter_name: str, metric_name: str) -> Dict[str, Any]:
    """Query one governed fighter metric from DuckDB.

    Args:
        fighter_name: Fighter name as stored in DuckDB.
        metric_name: Profile metric key (e.g. "punch_power").

    Returns:
        The metric payload, or {"status": "error", "error": "..."} on failure.
    """
    try:
        return tools.query_semantic_metric(fighter_name, metric_name)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}


@mcp.tool()
def run_stochastic_simulation(fighter_a: str, fighter_b: str, num_simulations: int = 1000) -> Dict[str, Any]:
    """Run N Monte Carlo bouts under Gaussian damage volatility.

    Args:
        fighter_a: First fighter name.
        fighter_b: Second fighter name.
        num_simulations: Monte Carlo sample count.

    Returns:
        Win/draw rates and method breakdown, or {"status": "error", "error": "..."}.
    """
    try:
        return tools.run_stochastic_simulation(fighter_a, fighter_b, num_simulations)
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}


if __name__ == "__main__":
    mcp.run()
