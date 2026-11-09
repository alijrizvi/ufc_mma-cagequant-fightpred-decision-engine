import pytest
from src.agent.tools import query_semantic_metric, run_stochastic_simulation
from src.agent.auditor import AnalyticalAuditor
from src.agent.copilot import AgenticCopilot

def test_fastmcp_tools_return_structured_dict():
    """Enforces that FastMCP analytical tools return structured ground-truth dictionaries."""
    metric_res = query_semantic_metric("Islam Makhachev", "punch_power")
    assert metric_res["fighter_name"] == "Islam Makhachev"
    assert "value" in metric_res

    sim_res = run_stochastic_simulation("Islam Makhachev", "Carlos Prates", num_simulations=50)
    assert sim_res["fighter_a"] == "Islam Makhachev"
    assert 0.0 <= sim_res["win_rate_a"] <= 1.0

def test_analytical_auditor_verification():
    """Enforces that AnalyticalAuditor validates numeric claims within a 1.5% tolerance window."""
    auditor = AnalyticalAuditor()
    ground_truth = {"win_rate_a": 0.65, "win_rate_b": 0.35}
    
    # Valid claim matching ground truth
    valid_text = "Islam Makhachev is favored with a 65.0% expected win rate."
    is_valid, violations = auditor.verify_claims(valid_text, ground_truth)
    assert is_valid is True
    assert len(violations) == 0

    # Hallucinated claim outside tolerance
    hallucinated_text = "Islam Makhachev holds an 88.0% expected win rate."
    is_valid_h, violations_h = auditor.verify_claims(hallucinated_text, ground_truth)
    assert is_valid_h is False
    assert len(violations_h) > 0

def test_copilot_mocked_execution():
    """Enforces that AgenticCopilot executes orchestration loop and passes auditor check."""
    copilot = AgenticCopilot(use_mock=True)
    brief = copilot.generate_tactical_brief("Islam Makhachev", "Carlos Prates")
    
    assert isinstance(brief.summary, str)
    assert brief.auditor_passed is True