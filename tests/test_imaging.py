import pytest
from PIL import Image

from src.imaging.pipeline import MedicalImagingPipeline
from src.imaging.benchmarker import ImagingArchitectureBenchmarker

def test_imaging_pipeline_tasks():
    pipeline = MedicalImagingPipeline()
    img = Image.new("RGB", (200, 200), color="grey")
    
    # 1. Classification
    c_res = pipeline.classify(img, "mri")
    assert c_res["diagnosis"] == "Glioblastoma Multiforme (Brain Tumor)"
    assert c_res["confidence"] > 0.80
    
    # 2. Segmentation
    s_res = pipeline.segment(img, "wound")
    assert len(s_res["lesion_mask_polygon"]) == 4
    assert s_res["segmented_area_percent"] == 28.2
    
    # 3. Detection & Localization
    d_res = pipeline.detect(img, "x-ray")
    assert len(d_res["bounding_box"]) == 4
    
    l_res = pipeline.localize(img, "x-ray")
    assert len(l_res["centroid"]) == 2
    
    # 4. Severity Estimation
    v_res = pipeline.estimate_severity(img, "histopathology")
    assert v_res["severity_level"] == "Severe"
    assert v_res["severity_score"] == 0.92

def test_imaging_pipeline_sanitization():
    pipeline = MedicalImagingPipeline()
    img = Image.new("RGB", (200, 200), color="grey")
    
    # Invalid modality fallback
    c_res = pipeline.classify(img, "invalid_scan")
    assert c_res["modality"] == "x-ray"
    assert "Pneumonia" in c_res["diagnosis"]

def test_imaging_benchmarker():
    benchmarker = ImagingArchitectureBenchmarker()
    benchmarks = benchmarker.run_comparative_benchmark()
    
    assert "Med-ViT" in benchmarks
    assert "UNet" in benchmarks
    assert "Florence-2" in benchmarks
    assert benchmarks["UNet"]["segmentation_iou"] == 0.895
    assert benchmarks["BioMedCLIP"]["classification_accuracy"] == 0.960
