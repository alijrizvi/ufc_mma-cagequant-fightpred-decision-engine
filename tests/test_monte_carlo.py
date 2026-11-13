import pytest

from src.engine.monte_carlo import MonteCarloRiskEngine, MonteCarloSummary
from src.engine.stats import FighterCombatProfile


def test_monte_carlo_batch_simulation():
    """Enforces that Monte Carlo engine executes N=50 iterations and outputs bounded probability distributions."""
    profile_a = FighterCombatProfile(name="Islam Makhachev", punch_power=24.0, arm_block_pct=0.45)
    profile_b = FighterCombatProfile(name="Carlos Prates", punch_power=26.0, arm_block_pct=0.35)

    mc_engine = MonteCarloRiskEngine(num_simulations=50)
    summary = mc_engine.run_bout_simulation(profile_a, profile_b, max_rounds=3)

    assert isinstance(summary, MonteCarloSummary)
    assert summary.total_simulations == 50
    assert 0.0 <= summary.win_rate_a <= 1.0
    assert 0.0 <= summary.win_rate_b <= 1.0
    assert summary.win_rate_a + summary.win_rate_b + summary.draw_rate == pytest.approx(1.0)
    assert summary.avg_finish_interval > 0.0