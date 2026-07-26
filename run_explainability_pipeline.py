import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.explainability.lab import MedicalFailureAnalysisLab
from src.explainability.explain import ClinicalPredictionExplainer

def generate_explainability_dashboard(shap_vals: dict, gradcam: dict, output_path: str = "docs/explainability_saliency.png"):
    """
    Plots game-theoretic SHAP vital attributions and Grad-CAM visual heat points.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Plot SHAP Vital attributions
    vitals = list(shap_vals.keys())
    scores = [shap_vals[v] for v in vitals]
    colors = ['tomato' if s > 0 else 'cornflowerblue' for s in scores]
    
    axes[0].barh(vitals, scores, color=colors)
    axes[0].set_xlabel('SHAP Attribution Weight')
    axes[0].set_title('Tabular Feature Attributions (SHAP)')
    axes[0].grid(axis='x', linestyle='--')
    
    # 2. Plot Grad-CAM heatmap grid points
    coords = gradcam["heatmap_grid"]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    weights = [c[2] * 200 for c in coords]
    
    axes[1].scatter(xs, ys, s=weights, c='red', alpha=0.6, edgecolors='black', label='Model Focus')
    axes[1].set_xlim(0, 256)
    axes[1].set_ylim(0, 256)
    axes[1].set_xlabel('Scan Coordinate X')
    axes[1].set_ylabel('Scan Coordinate Y')
    axes[1].set_title('Visual Saliency Map Points (Grad-CAM)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Explainability visual dashboard saved to: {output_path}")


def run_explainability_pipeline():
    print("="*75)
    print(" MEDTWIN AI: FAILURE ANALYSIS & EXPLAINABILITY LABORATORY ")
    print("="*75)
    
    # 1. Initialize Modules
    print("[1/4] Initializing failure analysis lab and explainers...")
    lab = MedicalFailureAnalysisLab()
    explainer = ClinicalPredictionExplainer()
    
    # 2. Run Failure Audit
    print("[2/4] Executing clinical failure audit on case file (False Negative)...")
    sample_case = {
        "prediction": "Normal",
        "ground_truth": "Pneumonia",
        "ocr_text": "Rx: Ceftriaxone 1g IV daily.",
        "missing_evidence": False
    }
    eval_metrics = {
        "rag_similarity": 0.85,
        "ner_faithfulness": 0.95
    }
    
    audit_res = lab.audit_prediction(sample_case, eval_metrics)
    print("\n" + "-"*65)
    print(" FAILURE ANALYSIS AUDIT LOG ")
    print("-"*65)
    print(f"  * Failure Detected:     {audit_res['failure_detected']}")
    print(f"  * Failure Classification: {audit_res['failure_class']}")
    print(f"  * Root Cause:           \"{audit_res['root_cause_analysis']}\"")
    print(f"  * Suggested Retraining:  \"{audit_res['suggested_retraining_action']}\"")
    print(f"  * Suggested Augment:     \"{audit_res['suggested_augmentation_strategy']}\"")
    print("-"*65)
    
    # 3. Generate Explainability Maps
    print("\n[3/4] Generating prediction attributions for active comorbidity case...")
    mock_img = Image.new("RGB", (256, 256), color="grey")
    vitals = {
        "temperature": 101.5,
        "wbc_count": 14500,
        "oxygen_saturation": 92.0
    }
    note = "Patient has severe cough, high fever, and chest pain."
    
    cam = explainer.generate_grad_cam(mock_img)
    ig = explainer.generate_integrated_gradients([0.2, 0.5, 0.8])
    shap = explainer.generate_shap_values(vitals)
    att = explainer.generate_attention_map(note)
    narrative = explainer.generate_natural_language_explanation("Pneumonia", 0.94)
    
    print("\n" + "-"*65)
    print(" EXPLAINABILITY ATTRIBUTION SUMMARY ")
    print("-"*65)
    print(f"  * Grad-CAM Target Layer:   {cam['target_layer']}")
    print(f"  * SHAP Vitals Attributions: {shap}")
    print(f"  * Integrated Attributions:  {ig['integrated_attributions']}")
    print(f"  * Self-Attention Tokens:    {list(att['attention_weights'].keys())}")
    print(f"\n{narrative}")
    print("-"*65)
    
    # 4. Export Dashboard Saliency
    print("\n[4/4] Plotting visual explainability dashboard...")
    generate_explainability_dashboard(shap, cam, "docs/explainability_saliency.png")
    
    print("\nExplainability Laboratory pipeline execution complete.")

if __name__ == "__main__":
    run_explainability_pipeline()
