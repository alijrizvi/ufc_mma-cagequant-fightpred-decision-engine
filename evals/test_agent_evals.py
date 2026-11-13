"""
Pytest integration test executing the 20 Golden Evaluation Scenarios.
"""
import json
from pathlib import Path

import pytest

from src.agent.copilot import DecisionCopilot


def load_scenarios():
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r") as f:
        return json.load(f)


@pytest.mark.parametrize("scenario", load_scenarios(), ids=lambda s: s["id"])
def test_copilot_golden_scenarios(scenario):
    copilot = DecisionCopilot(use_mock=True)
    result = copilot.execute_fight_analysis(
        fighter_a=scenario["fighter_a"],
        fighter_b=scenario["fighter_b"],
        user_query=scenario.get("query"),
        force_hallucination=scenario.get("force_hallucination", False),
        num_simulations=scenario.get("num_simulations", 1000),
    )
    assert result.status == scenario["expected_status"], (
        f"Scenario {scenario['id']} ({scenario['type']}) failed: "
        f"expected {scenario['expected_status']}, got {result.status} ({result.narrative})"
    )
