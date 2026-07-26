import pytest
from src.rag.medical_kg import MedicalKnowledgeGraph
from src.rag.vector_rag import MedicalRAG

def test_medical_knowledge_graph():
    kg = MedicalKnowledgeGraph()
    
    # Verify relations
    symptoms = kg.query_relations("Pneumonia", "has_symptom")
    assert "fever" in symptoms
    assert "cough" in symptoms
    
    icd = kg.query_relations("Pneumonia", "has_icd10_code")
    assert icd == ["J18.9"]
    
    # Verify symptom matching
    note = "Patient has high fever and deep chest pain."
    # Pneumonia symptoms: cough, fever, dyspnea, crackles. Only "fever" is in note.
    # Match rate should be 1/4 = 0.25
    rate = kg.verify_symptoms_against_disease("Pneumonia", note)
    assert rate == 0.25

def test_medical_rag():
    rag = MedicalRAG()
    
    # Query matching Pneumonia
    results = rag.retrieve_guidelines("severe cough, lung opacity, fever, J18.9")
    assert len(results) == 1
    doc = results[0]
    assert doc["disease"] == "Pneumonia"
    assert "PMID: 31573350" in doc["citation"]
    
    # Query matching Stroke
    results_stroke = rag.retrieve_guidelines("stroke thrombolysis window, NIHSS")
    assert len(results_stroke) == 1
    doc_stroke = results_stroke[0]
    assert doc_stroke["disease"] == "Acute Stroke"
    assert "AHA/ASA Ischemic Stroke" in doc_stroke["citation"]
