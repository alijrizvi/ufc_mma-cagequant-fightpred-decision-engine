import pytest
import json
import os
from src.agent.copilot import AgenticCopilot

def test_golden_eval_dataset_benchmark():
    """Evaluates AgenticCopilot performance across golden benchmark scenario prompts."""
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, "r") as f:
        golden_cases = json.load(f)

    copilot = AgenticCopilot(use_mock=True)
    
    passed_count = 0
    for case in golden_cases:
        brief = copilot.generate_tactical_brief(case["fighter_a"], case["fighter_b"])
        if brief.auditor_passed:
            passed_count += 1

    accuracy_rate = passed_count / len(golden_cases)
    assert accuracy_rate >= 0.90