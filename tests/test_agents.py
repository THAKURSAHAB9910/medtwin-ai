import pytest
from src.agents.state import ClinicalConsensusState, SpecialistOpinion
from src.agents.specialists import RadiologistAgent, CardiologistAgent
from src.agents.coordinator import CoordinatorAgent

def test_specialist_agent():
    radiologist = RadiologistAgent()
    patient = {
        "clinical_note": "Normal lungs.",
        "lab_metrics": {"temperature": 98.6}
    }
    opinion = radiologist.analyze(patient)
    assert isinstance(opinion, SpecialistOpinion)
    assert opinion.role == "Radiologist"
    assert opinion.diagnosis in ["Normal", "Pneumonia"]
    assert 0.0 <= opinion.confidence <= 1.0

def test_coordinator_consensus_voting():
    coordinator = CoordinatorAgent()
    
    # Patient with Pneumonia symptoms
    patient = {
        "clinical_note": "Cough, high fever. Lung haziness on scans.",
        "lab_metrics": {"temperature": 102.5, "wbc_count": 14000.0, "oxygen_saturation": 91.0}
    }
    
    state = coordinator.run_consensus(patient)
    assert isinstance(state, ClinicalConsensusState)
    assert len(state.specialist_opinions) == 7
    
    res = state.coordinator_consensus
    assert res["consensus_diagnosis"] == "Pneumonia"
    assert res["agreement_rate"] > 0.5
    assert res["consensus_uncertainty"] < 0.5
    
    # Valid tests compiled from concurrences should contain High-Resolution CT Thorax
    assert "High-Resolution CT Thorax" in res["final_tests"]
    # Treatment should contain antibiotic suggested by concurrences
    assert any("antibiotic" in treat.lower() for treat in res["final_treatments"])
