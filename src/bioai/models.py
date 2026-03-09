"""Shared data models — the contract between agents and the orchestrator."""

from enum import Enum
from typing import Any, Literal

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

class TranscriptomicsRecommendation(str, Enum):
    PHARMACOLOGY = "pharmacology"
    HEALTH_TRAINER = "health_trainer"


class TranscriptomicsFindings(BaseModel):
    pathway_scores: dict[str, float]
    dominant_pathway: str
    active_pathways: list[str]
    risk_level: RiskLevel
    dysregulated_genes: list[dict[str, Any]]
    diabetes_confirmed: dict[str, Any]
    diabetes_subtype: dict[str, str]
    complication_risks: list[dict[str, str]]
    monitoring: dict[str, Any]
    recommendation: TranscriptomicsRecommendation
    interpretation: str


class ProteomicsFindings(BaseModel):
    biomarker_scores: dict[str, float]
    elevated_biomarkers: list[str]
    biomarker_panel: str
    risk_level: RiskLevel
    complication_evidence: list[dict[str, str]]
    diabetes_confirmed: dict[str, Any]
    interpretation: str


class MetabolomicsFindings(BaseModel):
    metabolite_scores: dict[str, float]
    elevated_metabolites: list[str]
    insulin_resistance_score: float
    metabolic_pattern: str
    risk_level: RiskLevel
    subtype_refinement: dict[str, str]
    diabetes_confirmed: dict[str, Any]
    interpretation: str


class HospitalRecommendation(str, Enum):
    PHARMACOLOGY = "pharmacology"
    HEALTH_TRAINER = "health_trainer"


class HospitalFindings(BaseModel):
    patient_consented: bool
    transcriptomics_confirmed: bool
    metabolomics_confirmed: bool
    diabetes_confirmed: bool
    confidence: str  # "high" | "moderate" | "low"
    recommendation: HospitalRecommendation
    transcriptomics_summary: dict[str, Any]
    metabolomics_summary: dict[str, Any]
    reasoning: str


class PharmacologyFindings(BaseModel):
    diabetes_subtype: str
    primary_medications: list[dict[str, Any]]
    supportive_medications: list[dict[str, Any]]
    monitoring_plan: str
    medication_summary: str


class HealthTrainerFindings(BaseModel):
    fitness_level: Literal["beginner", "intermediate", "advanced"]
    goals: list[str]
    recommended_exercises: list[dict[str, Any]]
    weekly_plan: str


class AgentResult(BaseModel):
    agent: str
    status: AgentStatus
    findings: GenomicsFindings | DoctorFindings | TranscriptomicsFindings | ProteomicsFindings | MetabolomicsFindings | HospitalFindings | PharmacologyFindings | HealthTrainerFindings | None = None
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
