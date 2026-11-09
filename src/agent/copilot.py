from dataclasses import dataclass
from typing import Dict, Any, List
from src.agent.tools import run_stochastic_simulation
from src.agent.auditor import AnalyticalAuditor

@dataclass
class VerifiedTacticalBrief:
    summary: str
    ground_truth: Dict[str, Any]
    auditor_passed: bool
    violations: List[str]

class AgenticCopilot:
    """Orchestrates analytics tool invocation, response synthesis, and auditor verification."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.auditor = AnalyticalAuditor()

    def generate_tactical_brief(self, fighter_a: str, fighter_b: str) -> VerifiedTacticalBrief:
        """Runs stochastic simulations and synthesizes verified tactical briefs."""
        sim_data = run_stochastic_simulation(fighter_a, fighter_b, num_simulations=100)
        
        win_pct_a = round(sim_data["win_rate_a"] * 100.0, 1)
        win_pct_b = round(sim_data["win_rate_b"] * 100.0, 1)

        response_text = (
            f"Tactical Assessment: {fighter_a} holds an expected win rate of {win_pct_a}%, "
            f"while {fighter_b} maintains a {win_pct_b}% chance based on Monte Carlo risk simulation."
        )

        is_passed, violations = self.auditor.verify_claims(response_text, sim_data)

        return VerifiedTacticalBrief(
            summary=response_text,
            ground_truth=sim_data,
            auditor_passed=is_passed,
            violations=violations
        )