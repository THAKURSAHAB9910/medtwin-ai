import pytest
import numpy as np

from src.nlp.embeddings import MedicalEmbeddingGenerator
from src.nlp.vector_db import LocalVectorDB
from src.nlp.entity_extractor import ClinicalNER
from src.nlp.timeline import ClinicalTimelineCompiler
from src.nlp.rag_engine import ClinicalRAGEngine

def test_medical_embeddings():
    embedder = MedicalEmbeddingGenerator(dimension=128)
    
    emb1 = embedder.get_embedding("Patient presents with severe cough and pyrexia.")
    emb2 = embedder.get_embedding("Cough and fever symptoms reported.")
    emb3 = embedder.get_embedding("Standard cardiac chest pain protocols.")
    
    assert emb1.shape[0] == 128
    # Vitals embeddings should have cosine similarities greater than unrelated cardiac embeddings
    sim_related = np.dot(emb1, emb2)
    sim_unrelated = np.dot(emb1, emb3)
    
    assert sim_related > sim_unrelated

def test_local_vector_db():
    embedder = MedicalEmbeddingGenerator(dimension=128)
    db = LocalVectorDB(embedder)
    
    db.add_document("Ceftriaxone 1g IV daily is indicated for severe pneumonia.", {"type": "guideline"})
    db.add_document("Lisinopril 10mg daily is indicated for hypertension.", {"type": "guideline"})
    
    results = db.search("recommend medication for high blood pressure", k=1)
    assert len(results) == 1
    assert "Lisinopril" in results[0]["text"]
    assert results[0]["metadata"]["type"] == "guideline"

def test_clinical_ner_and_interactions():
    ner = ClinicalNER()
    note = "Patient taking Aspirin 325mg and Metformin 1000mg. Reports cough and fever."
    
    entities = ner.extract_entities(note)
    assert "Aspirin" in entities["medications"]
    assert "Metformin" in entities["medications"]
    assert "cough" in entities["symptoms"]
    assert "fever" in entities["symptoms"]
    assert "325mg" in entities["dosages"]
    
    # Test drug interactions
    meds = ["Aspirin", "Ibuprofen"]
    warnings = ner.check_drug_interactions(meds)
    assert len(warnings) == 1
    assert warnings[0]["severity"] == "Moderate"

def test_timeline_sorting():
    compiler = ClinicalTimelineCompiler()
    events = [
        {"date": "2026-03-01", "type": "Prescription", "content": "Metformin 500mg daily."},
        {"date": "2026-01-15", "type": "Lab Report", "content": "A1C recorded at 8.2%."},
        {"date": "2026-02-10", "type": "Note", "content": "Recommended lifestyle changes."}
    ]
    
    sorted_events = compiler.compile_timeline(events)
    assert sorted_events[0]["date"] == "2026-01-15"
    assert sorted_events[1]["date"] == "2026-02-10"
    assert sorted_events[2]["date"] == "2026-03-01"

def test_rag_hallucination_evaluation():
    embedder = MedicalEmbeddingGenerator(dimension=128)
    db = LocalVectorDB(embedder)
    ner = ClinicalNER()
    rag = ClinicalRAGEngine(db, ner)
    
    # Index source guidelines
    db.add_document("Pneumonia treatments recommend Ceftriaxone 1g daily.", {"source": "ATS"})
    
    # Query with factual source coverage
    res1 = rag.query_clinical_rag("pneumonia guidelines")
    assert res1["evaluation_metrics"]["faithfulness_score"] > 0.8
    assert res1["evaluation_metrics"]["hallucination_rate"] < 0.2
    
    # Query forcing out-of-context drug recommendations (hallucination)
    empty_db = LocalVectorDB(embedder)
    rag_empty = ClinicalRAGEngine(empty_db, ner)
    res2 = rag_empty.query_clinical_rag("pneumonia guidelines")
    
    # Empty db causes hallucination of Ceftriaxone answer when source doc is missing
    assert res2["evaluation_metrics"]["hallucination_rate"] > 0.3
