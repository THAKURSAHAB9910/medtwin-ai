from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class SpecialistOpinion(BaseModel):
    """
    Structured opinion schema produced by each Specialist Agent.
    """
    role: str = Field(description="Medical specialty role of the agent")
    diagnosis: str = Field(description="Primary clinical diagnostic conclusion")
    confidence: float = Field(description="Calibrated diagnostic confidence score [0, 1]")
    evidence: str = Field(description="Synthesized clinical evidence justifying diagnosis")
    recommended_tests: List[str] = Field(default_factory=list, description="Recommended diagnostic evaluations")
    treatment_suggestions: List[str] = Field(default_factory=list, description="Initial therapeutic recommendations")

class ClinicalConsensusState(BaseModel):
    """
    Shared memory schema for the Multi-Agent LangGraph clinical routing.
    Passes patient inputs, gathers specialist opinions, and records coordinator consensus.
    """
    patient_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patient details: clinical_note, lab_metrics, history, prescription, wearable_data"
    )
    specialist_opinions: Dict[str, SpecialistOpinion] = Field(
        default_factory=dict,
        description="Gathers specialist opinions mapped by role"
    )
    coordinator_consensus: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final aggregated consensus decision, disagreements, and citations"
    )
    history_trace: List[str] = Field(
        default_factory=list,
        description="Execution trace logging agent activations"
    )
