import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.synthetic.engine import SyntheticMedicalDataEngine
from src.synthetic.evaluator import SyntheticDataEvaluator

def generate_synthetic_dashboard(evaluation: dict, output_path: str = "docs/synthetic_data_evaluation.png"):
    """
    Plots baseline vs augmented model performances under noise and rare diseases.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    stages = ["Baseline Model", "Synthetic-Augmented Model"]
    clean_f1s = [evaluation["baseline_model"]["clean_f1_score"] * 100, evaluation["augmented_model"]["clean_f1_score"] * 100]
    noisy_f1s = [evaluation["baseline_model"]["degraded_noise_f1_score"] * 100, evaluation["augmented_model"]["degraded_noise_f1_score"] * 100]
    rare_sens = [evaluation["baseline_model"]["rare_disease_sensitivity"] * 100, evaluation["augmented_model"]["rare_disease_sensitivity"] * 100]
    hallucinations = [evaluation["baseline_model"]["summary_hallucination_rate"] * 100, evaluation["augmented_model"]["summary_hallucination_rate"] * 100]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(stages))
    width = 0.35
    
    # Plot 1: Clean vs Noisy F1-Score
    axes[0].bar(x - width/2, clean_f1s, width, label='Clean Scan F1 (%)', color='lightskyblue')
    axes[0].bar(x + width/2, noisy_f1s, width, label='Noisy Scan F1 (%)', color='tomato')
    axes[0].set_ylabel('F1 Score (%)')
    axes[0].set_title('Model Generalization Under Scan Quality Degradation')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stages)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--')
    
    # Plot 2: Rare Disease Sensitivity vs Hallucination Rate
    axes[1].bar(x - width/2, rare_sens, width, label='Rare Disease Sensitivity (%)', color='mediumpurple')
    axes[1].bar(x + width/2, hallucinations, width, label='Hallucination Rate (%)', color='pink')
    axes[1].set_ylabel('Accuracy / Error Rate (%)')
    axes[1].set_title('Orphan Disease Ingest Sensitivity & Reliability')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stages)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Synthetic medical data evaluation dashboard saved to: {output_path}")


def run_synthetic_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTI-MODAL SYNTHETIC MEDICAL DATA PIPELINE ")
    print("="*75)
    
    # 1. Initialize Engine
    print("[1/4] Initializing synthetic image and text generators...")
    engine = SyntheticMedicalDataEngine()
    evaluator = SyntheticDataEvaluator()
    
    # 2. Generate Degraded Scans
    print("[2/4] Simulating image degradation models...")
    for quality in ["high", "low-quality", "blurred", "noisy"]:
        img = engine.generate_image("mri", "glioblastoma", quality=quality)
        print(f"  * Generated glioblastoma MRI scan. Quality: {quality.upper()} | Dimensions: {img.size}")
        
    # 3. Generate Rare Disease Clinical Documents
    print("\n[3/4] Synthesizing clinical text files for Alkaptonuria (Rare)...")
    disease = "Alkaptonuria"
    rx = engine.generate_prescription(disease)
    note = engine.generate_discharge_summary(disease)
    labs = engine.generate_lab_report(disease)
    history = engine.generate_patient_history(disease)
    
    print("\n" + "-"*65)
    print(f" SYNTHETIC CASE PROFILE: {disease.upper()} (RARE CONDITION) ")
    print("-"*65)
    print(f"  * Prescription:      \"{rx}\"")
    print(f"  * Laboratory Panel:  \"{labs}\"")
    print(f"  * Patient History:   \"{history}\"")
    print(f"  * Discharge Summary: \"{note}\"")
    print("-"*65)
    
    # 4. Run Downstream Performance Evaluations
    print("\n[4/4] Evaluating synthetic data augmentation impact...")
    evaluation = evaluator.evaluate_synthetic_impact()
    
    print("\n" + "="*80)
    print(" SYNTHETIC DATA ROBUSTNESS EVALUATION REPORT ")
    print("="*80)
    print(" METRIC EVALUATED                  | BASELINE MODEL | SYNTHETIC-AUGMENTED MODEL ")
    print("-"*80)
    print(f" Clean Scan F1-Score               | {evaluation['baseline_model']['clean_f1_score']:14.1%} | {evaluation['augmented_model']['clean_f1_score']:25.1%} ")
    print(f" Degraded Noisy F1-Score           | {evaluation['baseline_model']['degraded_noise_f1_score']:14.1%} | {evaluation['augmented_model']['degraded_noise_f1_score']:25.1%} ")
    print(f" Rare Disease Sensitivity          | {evaluation['baseline_model']['rare_disease_sensitivity']:14.1%} | {evaluation['augmented_model']['rare_disease_sensitivity']:25.1%} ")
    print(f" QA Summary Hallucination Rate     | {evaluation['baseline_model']['summary_hallucination_rate']:14.1%} | {evaluation['augmented_model']['summary_hallucination_rate']:25.1%} ")
    print("="*80)
    
    generate_synthetic_dashboard(evaluation, "docs/synthetic_data_evaluation.png")
    print("\nSynthetic Medical Data pipeline execution complete.")

if __name__ == "__main__":
    run_synthetic_pipeline()
