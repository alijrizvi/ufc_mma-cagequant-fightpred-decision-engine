import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List
from src.engine.stats import FighterCombatProfile
from src.engine.simulation import OctagonSimulationEngine

# Batch Simulations (n = 1,000) for any Fight Matchup to determine
# the probability (distribution %s) of each fighter winning
# the probability of each method of victory (KO/TKO, Submission, Decision)
# the confidence intervals for the fight duration

@dataclass
class MonteCarloSummary:
    total_simulations: int
    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    method_breakdown: Dict[str, float]
    avg_finish_interval: float
    interval_ci_lower: float
    interval_ci_upper: float

class MonteCarloRiskEngine:
    """Quantitative risk engine executing N parallel bout simulations for probability distribution estimates."""

    def __init__(self, num_simulations: int = 100):
        self.num_simulations = num_simulations
        self.sim_engine = OctagonSimulationEngine()

    def run_bout_simulation(
        self, 
        profile_a: FighterCombatProfile, 
        profile_b: FighterCombatProfile, 
        max_rounds: int = 5
    ) -> MonteCarloSummary:
        """Executes N iterations of a matchup and aggregates win distributions and confidence intervals."""
        wins_a = 0
        wins_b = 0
        draws = 0
        
        methods: Dict[str, int] = {}
        finish_intervals: List[int] = []

        for _ in range(self.num_simulations):
            result = self.sim_engine.run_full_fight(profile_a, profile_b, max_rounds=max_rounds)
            
            # 1. Track Winners
            if result.winner == profile_a.name:
                wins_a += 1
            elif result.winner == profile_b.name:
                wins_b += 1
            else:
                draws += 1

            # 2. Track Method of Victory
            base_method = result.method_of_victory.split(" (")[0]
            methods[base_method] = methods.get(base_method, 0) + 1

            # 3. Track Duration
            finish_intervals.append(result.total_intervals)

        # Compute summary distribution metrics
        n = float(self.num_simulations)
        win_rate_a = round(wins_a / n, 4)
        win_rate_b = round(wins_b / n, 4)
        draw_rate = round(draws / n, 4)

        method_breakdown = {k: round(v / n, 4) for k, v in methods.items()}

        arr_intervals = np.array(finish_intervals, dtype=np.float32)
        avg_finish = float(np.mean(arr_intervals))
        ci_lower = float(np.percentile(arr_intervals, 2.5))
        ci_upper = float(np.percentile(arr_intervals, 97.5))

        return MonteCarloSummary(
            total_simulations=self.num_simulations,
            win_rate_a=win_rate_a,
            win_rate_b=win_rate_b,
            draw_rate=draw_rate,
            method_breakdown=method_breakdown,
            avg_finish_interval=round(avg_finish, 2),
            interval_ci_lower=round(ci_lower, 1),
            interval_ci_upper=round(ci_upper, 1)
        )