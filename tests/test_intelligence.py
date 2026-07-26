import pytest
from PIL import Image

from src.intelligence.clinical_fuser import MultimodalIntelligenceFuser
from src.intelligence.reasoning_engine import MultimodalReasoningEngine

def test_intelligence_aligned_fuser():
    fuser = MultimodalIntelligenceFuser()
    
    # Create simple mock radiograph image
    img = Image.new("RGB", (224, 224), color="grey")
    note = "Type 2 diabetes mellitus patient presents with chest discomfort."
    vitals = {"age": 55, "temperature": 98.6, "heart_rate": 84, "wbc_count": 8200, "oxygen_saturation": 98}
    timeline = "[2026-01-05] - NOTE: Initial clinical intake."
    genomics = {"EGFR_mutation": "mutated"}
    
    fused = fuser.fuse_patient_profile(
        image=img,
        clinical_note=note,
        vitals=vitals,
        timeline_narrative=timeline,
        genomic_data=genomics
    )
    
    assert fused["has_radiograph"] is True
    assert "Type 2 diabetes" in fused["narrative_text"]
    assert fused["vitals"]["wbc_count"] == 8200.0
    assert fused["genomics"]["EGFR_mutation"] == "mutated"

def test_intelligence_reasoning_engine():
    engine = MultimodalReasoningEngine()
    
    # Normal case
    fused_normal = {
        "narrative_text": "Patient checking vitals. No reports of disease.",
        "vitals": {"age": 30, "temperature": 98.4, "heart_rate": 72, "wbc_count": 6000, "oxygen_saturation": 99, "blood_pressure": 115, "blood_glucose": 95, "weight": 70},
        "genomics": {"EGFR_mutation": "not_detected", "BRCA1_mutation": "not_detected"},
        "has_radiograph": False
    }
    res_normal = engine.evaluate_profile(fused_normal)
    assert "Normal Health State" in res_normal["diagnosis"]
    assert res_normal["confidence_score"] > 0.90
    
    # Febrile Pneumonia and Diabetes Case
    fused_sick = {
        "narrative_text": "Patient with cough, chest consolidation opacities. Active history of diabetes mellitus.",
        "vitals": {"age": 68, "temperature": 101.5, "heart_rate": 96, "wbc_count": 13400, "oxygen_saturation": 91, "blood_pressure": 140, "blood_glucose": 182, "weight": 85},
        "genomics": {"EGFR_mutation": "mutated", "BRCA1_mutation": "not_detected"},
        "has_radiograph": True
    }
    res_sick = engine.evaluate_profile(fused_sick)
    
    assert "Community-Acquired Pneumonia" in res_sick["diagnosis"]
    assert "Type 2 Diabetes Mellitus" in res_sick["diagnosis"]
    assert "Essential Hypertension" in res_sick["diagnosis"]
    assert res_sick["confidence_score"] > 0.80
    assert len(res_sick["recommended_investigations"]) > 2
    
    # Verify genomic and hypoxia alerts are triggered
    risks_str = " ".join(res_sick["risk_factors"])
    assert "Genetic cancer risk" in risks_str
    assert "Hypoxia danger" in risks_str
    
    # Verify evidence logs temperature and white blood count
    assert "febrile" in res_sick["evidence"].lower()
    assert "leukocytosis" in res_sick["evidence"].lower()
