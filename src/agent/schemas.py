"""Pydantic contracts for the agent tools and the Copilot's structured outputs."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MetricQueryRequest(BaseModel):
    fighter_name: str = Field(..., description="Full legal name of the fighter.")
    metric_name: str = Field(..., description="Semantic metric key defined in metrics.yml (e.g. xCT, SDR, SVI).")


class MetricQueryResponse(BaseModel):
    fighter_name: str
    metric_name: str
    value: float
    status: str = "governed_success"


class SimulationRequest(BaseModel):
    fighter_a: str = Field(..., description="First fighter name.")
    fighter_b: str = Field(..., description="Second fighter name.")
    num_simulations: int = Field(default=1000, ge=100, le=10000, description="Monte Carlo sample count.")


class SimulationResponse(BaseModel):
    fighter_a: str
    fighter_b: str
    total_simulations: int
    win_rate_a: float
    win_rate_b: float
    draw_rate: float
    avg_finish_interval: float
    method_breakdown: Dict[str, float]


class CopilotAnalysisResult(BaseModel):
    status: str  # "success", "refused", or "error"
    narrative: str
    violations: List[str] = Field(default_factory=list)
    ground_truth: Optional[Dict] = None
    attempts: int = 0  # narration attempts made by the harness
    trace: List[str] = Field(default_factory=list)  # plain-English audit trail of each harness step
