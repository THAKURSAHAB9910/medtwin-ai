import pytest
from PIL import Image

from src.explainability.lab import MedicalFailureAnalysisLab
from src.explainability.explain import ClinicalPredictionExplainer

def test_failure_analysis_lab():
    lab = MedicalFailureAnalysisLab()
    
    # 1. Test False Negative
    c1 = {"prediction": "Normal", "ground_truth": "Pneumonia"}
    res1 = lab.audit_prediction(c1, {})
    assert res1["failure_detected"] is True
    assert res1["failure_class"] == "False Negative"
    assert "focal loss" in res1["suggested_retraining_action"]
    
    # 2. Test False Positive
    c2 = {"prediction": "Pneumonia", "ground_truth": "Normal"}
    res2 = lab.audit_prediction(c2, {})
    assert res2["failure_class"] == "False Positive"
    
    # 3. Test Hallucination
    c3 = {"prediction": "Pneumonia", "ground_truth": "Pneumonia"}
    metrics3 = {"ner_faithfulness": 0.50}
    res3 = lab.audit_prediction(c3, metrics3)
    assert res3["failure_class"] == "Hallucination"
    
    # 4. Test Retrieval Failure
    metrics4 = {"ner_faithfulness": 1.0, "rag_similarity": 0.20}
    res4 = lab.audit_prediction(c3, metrics4)
    assert res4["failure_class"] == "Retrieval Failure"

def test_explainers():
    explainer = ClinicalPredictionExplainer()
    img = Image.new("RGB", (200, 200), color="grey")
    
    # 1. Grad-CAM
    cam = explainer.generate_grad_cam(img)
    assert len(cam["heatmap_grid"]) == 4
    
    # 2. SHAP
    shap = explainer.generate_shap_values({"temperature": 101.2, "wbc_count": 5000})
    assert shap["temperature"] == 0.35
    assert shap["wbc_count"] == 0.05
    
    # 3. Integrated Gradients
    ig = explainer.generate_integrated_gradients([1.0, 2.0])
    assert len(ig["integrated_attributions"]) == 2
    
    # 4. Attentions
    att = explainer.generate_attention_map("Cough and fever indicators")
    assert att["attention_weights"]["Cough"] == 0.82
