"""Planner -> Executor -> Auditor harness.

A deliberately light "System 2" loop (see ADR-014: no framework dependency).
Every run leaves a plain-English `trace`, so a human can audit what the agent did.

    plan()    decides: refuse, or simulate  (deterministic rules, no LLM)
    execute() calls the governed tool       (ground truth comes only from here)
    run()     narrates -> audits -> retries with feedback -> answers or refuses
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.agent.auditor import AnalyticalAuditor
from src.agent.schemas import CopilotAnalysisResult
from src.agent.tools import run_stochastic_simulation

REFUSAL_PATTERN = re.compile(
    r"\b(who\s+will\s+win|pick\s+the\s+winner|bet|gamble|wager|odds)\b",
    re.IGNORECASE,
)

# narrator(ground_truth, fighter_a, fighter_b, prior_violations) -> draft text
Narrator = Callable[[dict[str, Any], str, str, list[str]], str]


@dataclass(frozen=True)
class Plan:
    """What the Planner decided to do."""

    action: str  # "simulate" | "refuse"
    fighter_a: str
    fighter_b: str
    num_simulations: int = 1000
    reason: str = ""


class AgentHarness:
    """Runs one analysis request through plan -> execute -> narrate -> audit."""

    def __init__(
        self,
        narrator: Narrator,
        auditor: AnalyticalAuditor | None = None,
        max_attempts: int = 2,
        simulate: Callable[..., dict[str, Any]] = run_stochastic_simulation,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        self.narrator = narrator
        self.auditor = auditor or AnalyticalAuditor()
        self.max_attempts = max_attempts
        self.simulate = simulate

    def plan(self, fighter_a: str, fighter_b: str, user_query: str | None, num_simulations: int = 1000) -> Plan:
        """Planner: refuse betting-style questions, otherwise simulate."""
        if user_query and REFUSAL_PATTERN.search(user_query):
            return Plan("refuse", fighter_a, fighter_b, num_simulations, "betting/pick-the-winner request")
        return Plan("simulate", fighter_a, fighter_b, num_simulations)

    def execute(self, plan: Plan) -> dict[str, Any]:
        """Executor: the only place ground-truth numbers are produced.

        Raises:
            ValueError: If a fighter is unknown (bubbles up to run()).
        """
        return self.simulate(plan.fighter_a, plan.fighter_b, num_simulations=plan.num_simulations)

    def run(
        self,
        fighter_a: str,
        fighter_b: str,
        user_query: str | None = None,
        num_simulations: int = 1000,
    ) -> CopilotAnalysisResult:
        trace: list[str] = []
        plan = self.plan(fighter_a, fighter_b, user_query, num_simulations)
        trace.append(f"PLAN: {plan.action}" + (f" ({plan.reason})" if plan.reason else ""))

        if plan.action == "refuse":
            return CopilotAnalysisResult(
                status="refused",
                narrative=(
                    "Refusal: CageQuant models stochastic win distributions and telemetry risk, "
                    "not gambling advice or definitive picks."
                ),
                trace=trace,
            )

        try:
            ground_truth = self.execute(plan)
        except ValueError as exc:
            trace.append(f"EXECUTE: failed ({exc})")
            return CopilotAnalysisResult(status="error", narrative=f"Cannot analyze this matchup: {exc}", trace=trace)
        trace.append(f"EXECUTE: simulated {plan.num_simulations} bouts")

        violations: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                draft = self.narrator(ground_truth, fighter_a, fighter_b, violations)
            except Exception as exc:  # LLM/network failures must not crash the app
                trace.append(f"NARRATE #{attempt}: failed ({type(exc).__name__})")
                return CopilotAnalysisResult(
                    status="error",
                    narrative="The narration service is unavailable; the raw simulation is still valid.",
                    ground_truth=ground_truth,
                    attempts=attempt,
                    trace=trace,
                )
            is_valid, violations = self.auditor.verify_claims(draft, ground_truth)
            trace.append(f"AUDIT #{attempt}: {'pass' if is_valid else f'fail ({len(violations)} violation(s))'}")
            if is_valid:
                return CopilotAnalysisResult(
                    status="success",
                    narrative=draft,
                    ground_truth=ground_truth,
                    attempts=attempt,
                    trace=trace,
                )

        return CopilotAnalysisResult(
            status="refused",
            narrative="Output refused: Narrative failed audit verification due to uncalibrated or hallucinated statistics.",
            violations=violations,
            ground_truth=ground_truth,
            attempts=self.max_attempts,
            trace=trace,
        )
