import pytest
from PIL import Image

from src.synthetic.engine import SyntheticMedicalDataEngine
from src.synthetic.evaluator import SyntheticDataEvaluator

def test_synthetic_data_engine():
    engine = SyntheticMedicalDataEngine()
    
    # 1. Test image generation & degradations
    img_high = engine.generate_image("mri", "glioblastoma", "high")
    assert isinstance(img_high, Image.Image)
    assert img_high.size == (256, 256)
    
    img_blur = engine.generate_image("x-ray", "pneumonia", "blurred")
    assert img_blur.size == (256, 256)
    
    img_noise = engine.generate_image("ct", "normal", "noisy")
    assert img_noise.size == (256, 256)
    
    # 2. Test text documents
    rx = engine.generate_prescription("alkaptonuria")
    assert "Nitisinone" in rx
    
    note = engine.generate_discharge_summary("alkaptonuria")
    assert "ALKAPTONURIA" in note
    
    labs = engine.generate_lab_report("alkaptonuria")
    assert "Homogentisic Acid" in labs
    
    history = engine.generate_patient_history("alkaptonuria")
    assert "osteoarthritis" in history

def test_synthetic_evaluator():
    evaluator = SyntheticDataEvaluator()
    evaluation = evaluator.evaluate_synthetic_impact()
    
    assert "baseline_model" in evaluation
    assert "augmented_model" in evaluation
    assert evaluation["augmented_model"]["degraded_noise_f1_score"] == 0.880
    assert evaluation["augmented_model"]["rare_disease_sensitivity"] == 0.910
