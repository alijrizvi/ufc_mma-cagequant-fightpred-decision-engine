"""Decision Copilot: the narration layer on top of the AgentHarness."""

from __future__ import annotations

import os
from typing import Any

from anthropic import Anthropic

from src.agent.auditor import AnalyticalAuditor
from src.agent.harness import REFUSAL_PATTERN, AgentHarness
from src.agent.schemas import CopilotAnalysisResult

DEFAULT_MODEL = "claude-haiku-5-5"  # override with the ANTHROPIC_MODEL env var


class DecisionCopilot:
    REFUSAL_PATTERN = REFUSAL_PATTERN  # kept for backward compatibility

    def __init__(self, tolerance_pct: float = 0.5, use_mock: bool = False, max_attempts: int = 2) -> None:
        self.use_mock = use_mock
        self.tolerance_pct = tolerance_pct
        self.max_attempts = max_attempts
        self.auditor = AnalyticalAuditor(tolerance_pct=tolerance_pct)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.model_name = os.getenv("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.client = Anthropic(api_key=self.api_key) if (self.api_key and not use_mock) else None

    def _narrator(self, force_hallucination: bool):
        """Builds the narrator callable the harness will use (mock or real LLM)."""

        def narrate(ground_truth: dict[str, Any], fighter_a: str, fighter_b: str, violations: list[str]) -> str:
            p_a = round(ground_truth["win_rate_a"] * 100, 1)
            p_b = round(ground_truth["win_rate_b"] * 100, 1)

            if self.use_mock or self.client is None:
                if force_hallucination:
                    return f"Tactical brief: {fighter_a} has an overwhelming 98.5% win rate over {fighter_b}."
                return (
                    f"Across {ground_truth.get('total_simulations', 1000):,} stochastic iterations, {fighter_a} registers a {p_a}% projected win rate, "
                    f"while {fighter_b} sits at {p_b}% with an average finish interval of "
                    f"{ground_truth['avg_finish_interval']}."
                )

            prompt = (
                f"Ground truth telemetry: {ground_truth}. Provide a 2-sentence tactical brief stating "
                f"exact win rates ({p_a}% vs {p_b}%). Do not hallucinate or alter any numbers."
            )
            if violations:
                prompt += f" Your previous draft failed an audit: {violations}. Use ONLY the numbers given."
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return narrate

    def execute_fight_analysis(
        self,
        fighter_a: str,
        fighter_b: str,
        user_query: str | None = None,
        force_hallucination: bool = False,
        num_simulations: int = 1000,
    ) -> CopilotAnalysisResult:
        harness = AgentHarness(
            narrator=self._narrator(force_hallucination),
            auditor=self.auditor,
            max_attempts=self.max_attempts,
        )
        return harness.run(fighter_a, fighter_b, user_query, num_simulations)


# Backward compatibility alias
AgenticCopilot = DecisionCopilot
