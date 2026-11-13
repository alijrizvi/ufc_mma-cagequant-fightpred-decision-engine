import pytest

from src.agent.harness import AgentHarness


def fake_sim(a, b, num_simulations=1000):
    return {"win_rate_a": 0.62, "win_rate_b": 0.38, "avg_finish_interval": 7.5}


def good_narrator(gt, a, b, violations):
    return f"{a} wins 62.0% of simulations; {b} wins 38.0%."


def test_planner_refuses_betting_questions():
    h = AgentHarness(good_narrator, simulate=fake_sim)
    assert h.plan("A", "B", "What are the odds?").action == "refuse"
    assert h.plan("A", "B", "Compare their grappling").action == "simulate"


def test_run_success_records_trace():
    result = AgentHarness(good_narrator, simulate=fake_sim).run("A", "B")
    assert result.status == "success"
    assert [t.split(":")[0] for t in result.trace] == ["PLAN", "EXECUTE", "AUDIT #1"]


def test_auditor_feedback_loop_recovers_on_second_attempt():
    calls = []

    def flaky(gt, a, b, violations):
        calls.append(list(violations))
        return "A has a 99.0% win rate." if not violations else "A wins 62.0% of simulations."

    result = AgentHarness(flaky, simulate=fake_sim, max_attempts=2).run("A", "B")
    assert result.status == "success" and result.attempts == 2
    assert calls[0] == [] and calls[1]  # second attempt received the violations


def test_persistent_hallucination_is_refused():
    def liar(gt, a, b, violations):
        return "A has a 99.0% win rate."

    result = AgentHarness(liar, simulate=fake_sim, max_attempts=2).run("A", "B")
    assert result.status == "refused" and result.violations


def test_unknown_fighter_returns_error_not_crash():
    def boom(a, b, num_simulations=1000):
        raise ValueError("Fighter 'X' not found in DuckDB database.")

    result = AgentHarness(good_narrator, simulate=boom).run("X", "Y")
    assert result.status == "error"


def test_narrator_outage_returns_error():
    def down(gt, a, b, violations):
        raise TimeoutError("api down")

    result = AgentHarness(down, simulate=fake_sim).run("A", "B")
    assert result.status == "error" and result.ground_truth is not None


def test_max_attempts_must_be_positive():
    with pytest.raises(ValueError):
        AgentHarness(good_narrator, max_attempts=0)
