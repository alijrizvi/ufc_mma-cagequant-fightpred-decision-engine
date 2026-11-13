"""
Deterministic Evaluation Runner benchmarking Copilot accuracy, refusals, and audit gates.
"""
import json
from pathlib import Path

from src.agent.copilot import DecisionCopilot


def run_evals():
    dataset_path = Path(__file__).parent / "golden_dataset.json"
    with open(dataset_path, "r") as f:
        scenarios = json.load(f)

    copilot = DecisionCopilot(use_mock=True, tolerance_pct=0.5)
    passed = 0

    print(f"\n--- Executing {len(scenarios)} Golden Evaluation Scenarios ---")
    for case in scenarios:
        case_id = case["id"]
        force_hal = case.get("force_hallucination", False)
        query = case.get("query", None)
        
        result = copilot.execute_fight_analysis(
            fighter_a=case["fighter_a"],
            fighter_b=case["fighter_b"],
            user_query=query,
            force_hallucination=force_hal,
            num_simulations=case.get("num_simulations", 100),
        )
        
        status_match = (result.status == case["expected_status"])
        if status_match:
            passed += 1
            print(f"[{case_id}] PASS - Status: {result.status}")
        else:
            print(f"[{case_id}] FAIL - Expected: {case['expected_status']}, Got: {result.status}")

    pass_rate = (passed / len(scenarios)) * 100
    print(f"\nEvaluation Benchmark Result: {passed}/{len(scenarios)} passed ({pass_rate:.1f}%)")
    assert pass_rate >= 90.0, f"Benchmark failed: pass rate {pass_rate}% below 90% threshold."

if __name__ == "__main__":
    run_evals()
