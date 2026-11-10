"""
Closed-loop Agentic Copilot with LLM routing and Receipt-Style Auditing.
"""
import os
from typing import Dict, Any, Optional
from anthropic import Anthropic
from src.agent.auditor import AnalyticalAuditor
from src.agent.mcp_server import run_stochastic_simulation
from src.agent.schemas import CopilotAnalysisResult

class DecisionCopilot:
    def __init__(self, tolerance_pct: float = 1.6, use_mock: bool = False):
        self.use_mock = use_mock
        self.tolerance_pct = tolerance_pct
        self.auditor = AnalyticalAuditor(tolerance_pct=tolerance_pct)
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = Anthropic(api_key=self.api_key) if (self.api_key and not use_mock) else None

    def execute_fight_analysis(
        self, 
        fighter_a: str, 
        fighter_b: str, 
        user_query: Optional[str] = None,
        force_hallucination: bool = False
    ) -> CopilotAnalysisResult:
        """
        Closed Loop: Intent Check -> FastMCP Execution -> LLM Narration -> Receipt Audit Gate.
        """
        # Guardrail / Refusal check: Refuse pure future predictions or financial/gambling advice
        if user_query and any(w in user_query.lower() for w in ["who will win", "bet", "pick the winner"]):
            return CopilotAnalysisResult(
                status="refused",
                narrative="Refusal: CageQuant models stochastic win distributions and telemetry risk, not gambling advice or definitive winner picks.",
                violations=[]
            )

        # 1. Execute deterministic FastMCP tool
        ground_truth: Dict[str, Any] = run_stochastic_simulation(
            fighter_a=fighter_a, 
            fighter_b=fighter_b, 
            num_simulations=1000
        )

        # 2. LLM Narration
        if self.use_mock or self.client is None:
            # Deterministic mock narration for zero-cost CI
            p_a = round(ground_truth["win_rate_a"] * 100, 1)
            p_b = round(ground_truth["win_rate_b"] * 100, 1)
            
            if force_hallucination:
                # Deliberate hallucination to test auditor gate in evals
                draft_narrative = f"Tactical brief: {fighter_a} has an overwhelming 98.5% win rate over {fighter_b}."
            else:
                draft_narrative = (
                    f"Across 1,000 stochastic iterations, {fighter_a} registers a {p_a}% projected win rate, "
                    f"while {fighter_b} sits at {p_b}% with an average finish interval of {ground_truth['avg_finish_interval']}."
                )
        else:
            system_prompt = (
                "You are a combat sports Decision Intelligence Copilot. Summarize the simulation receipt into a concise "
                "2-sentence tactical brief. State the win percentages directly. NEVER alter or fabricate numbers."
            )
            user_msg = f"Telemetry Simulation Receipt: {ground_truth}"
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_msg}]
            )
            draft_narrative = response.content[0].text

        # 3. Receipt-Style Audit Gate (Checks claims vs ground truth)
        is_valid, violations = self.auditor.verify_claims(draft_narrative, ground_truth)

        if not is_valid:
            return CopilotAnalysisResult(
                status="refused",
                narrative="Output refused: Narrative failed audit verification due to uncalibrated or hallucinated statistics.",
                violations=violations,
                ground_truth=ground_truth
            )

        return CopilotAnalysisResult(
            status="success",
            narrative=draft_narrative,
            violations=[],
            ground_truth=ground_truth
        )