"""Shared data models — the contract between agents and the orchestrator."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class RiskLevel(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


class AgentStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Per-agent findings
# ---------------------------------------------------------------------------

class GenomicsFindings(BaseModel):
    predicted_class: Literal["DMT1", "DMT2", "NONDM"]
    confidence: float
    probabilities: dict[str, float]
    risk_level: RiskLevel
    interpretation: str


class Recommendation(str, Enum):
    HOSPITAL = "hospital"
    HEALTH_TRAINER = "health_trainer"


class DoctorFindings(BaseModel):
    prediction: Literal["Diabetic", "Non-Diabetic"]
    probability: float
    risk_level: RiskLevel
    recommendation: Recommendation
    reasoning: str


# ---------------------------------------------------------------------------
# Generic agent result — wraps any agent's findings
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    agent: str
    status: AgentStatus
    findings: GenomicsFindings | DoctorFindings | None = None
    summary: str
    error: str | None = None


# ---------------------------------------------------------------------------
# Orchestrator output
# ---------------------------------------------------------------------------

class HealthAssessment(BaseModel):
    patient_id: str | None = None
    agent_results: list[AgentResult]
    overall_risk: RiskLevel | None = None
    report: str  # synthesized narrative from orchestrator
