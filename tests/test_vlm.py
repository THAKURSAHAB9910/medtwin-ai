import pytest
import os
import torch
from PIL import Image

from src.vlm.unified_api import MedicalVLMRegistry
from src.vlm.explain import VLMExplainer
from src.vlm.benchmark import VLMPerformanceBenchmarker

def test_medical_vlm_registry_switching():
    registry = MedicalVLMRegistry(use_mock=True)
    
    # Verify model switching
    assert registry.current_model == "qwen2-vl"
    registry.switch_model("biomedclip")
    assert registry.current_model == "biomedclip"
    
    with pytest.raises(ValueError):
        registry.switch_model("invalid-vlm-model")

def test_vlm_inference_outputs():
    registry = MedicalVLMRegistry(use_mock=True)
    img = Image.new("RGB", (100, 100), color="white")
    
    # 1. Qwen2-VL
    registry.switch_model("qwen2-vl")
    res_qwen = registry.infer(img, "explain scan", task="reasoning")
    assert res_qwen["model"] == "qwen2-vl"
    assert "conversational_response" in res_qwen["outputs"]
    
    # 2. BioMedCLIP
    registry.switch_model("biomedclip")
    res_clip = registry.infer(img, "cough fever", task="diagnosis")
    assert "contrastive_scores" in res_clip["outputs"]
    
    # 3. Florence-2
    registry.switch_model("florence-2")
    res_florence = registry.infer(img, "opacity", task="grounding")
    assert "grounded_box" in res_florence["outputs"]
    
    # 4. LayoutLMv3
    registry.switch_model("layoutlmv3")
    res_layout = registry.infer(img, "prescription", task="ocr")
    assert "extracted_entities" in res_layout["outputs"]
    assert "ocr_raw_text" in res_layout["outputs"]

def test_vlm_explainer_grounding():
    explainer = VLMExplainer(output_dir="docs")
    img = Image.new("RGB", (256, 256), color="black")
    bboxes = [[50, 50, 150, 150]]
    
    save_path = explainer.draw_grounded_boxes(img, bboxes, label="Test Anomaly", save_name="test_grounding_out.png")
    assert os.path.exists(save_path)
    assert save_path.endswith("test_grounding_out.png")
    
    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)

def test_vlm_performance_benchmarker():
    registry = MedicalVLMRegistry(use_mock=True)
    benchmarker = VLMPerformanceBenchmarker(registry)
    
    results = benchmarker.benchmark_all(dataloader=None)
    assert len(results) == 5
    for model in ["qwen2-vl", "llava-med", "biomedclip", "florence-2", "layoutlmv3"]:
        assert model in results
        assert "diagnosis_accuracy" in results[model]
        assert "ocr_cer" in results[model]
        assert "latency_ms" in results[model]
        assert "vram_mb" in results[model]
