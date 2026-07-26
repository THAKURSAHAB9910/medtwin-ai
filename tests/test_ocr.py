import pytest
from PIL import Image

from src.nlp.entity_extractor import ClinicalNER
from src.ocr.pipeline import MedicalOCRPipeline
from src.ocr.benchmarker import OCRArchitectureBenchmarker

def test_ocr_pipeline_extractions():
    ner = ClinicalNER()
    pipeline = MedicalOCRPipeline(ner)
    img = Image.new("RGB", (200, 100), color="white")
    
    # 1. Test Prescription OCR
    res1 = pipeline.extract_text_and_entities(img, "prescription")
    assert res1["document_type"] == "prescription"
    assert "Metformin" in res1["extracted_medications"]
    assert "500mg" in res1["extracted_dosages"]
    
    # 2. Test Lab Report OCR
    res2 = pipeline.extract_text_and_entities(img, "lab report")
    assert "165 mg" in res2["extracted_dosages"]
    assert "Diabetes" in res2["extracted_diseases"]

def test_ocr_pipeline_sanitization():
    ner = ClinicalNER()
    pipeline = MedicalOCRPipeline(ner)
    img = Image.new("RGB", (200, 100), color="white")
    
    # Fallback checks
    res = pipeline.extract_text_and_entities(img, "invalid_document")
    assert res["document_type"] == "prescription"
    assert "Metformin" in res["extracted_medications"]

def test_ocr_benchmarker():
    benchmarker = OCRArchitectureBenchmarker()
    benchmarks = benchmarker.run_comparative_benchmark()
    
    assert "PaddleOCR" in benchmarks
    assert "TrOCR" in benchmarks
    assert "Florence OCR" in benchmarks
    
    assert benchmarks["TrOCR"]["character_accuracy"] == 0.978
    assert benchmarks["EasyOCR"]["latency_ms"] == 22.0
