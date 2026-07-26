import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import MedicalDataModule
from src.agents.coordinator import CoordinatorAgent
from src.evaluation.consensus_evaluator import ConsensusEvaluator

def generate_consensus_plots(report: dict, output_path: str = "docs/consensus_benchmark_dashboard.png"):
    """
    Plots the comparative metrics between Single LLM and Multi-Agent Consensus.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Accuracy & F1 & Hallucination comparison
    metrics = ["Accuracy", "F1-Score", "Hallucination Rate"]
    b_scores = [
        report["baseline"]["accuracy"],
        report["baseline"]["f1"],
        report["baseline"]["hallucination_rate"]
    ]
    c_scores = [
        report["consensus"]["accuracy"],
        report["consensus"]["f1"],
        report["consensus"]["hallucination_rate"]
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0].bar(x - width/2, b_scores, width, label="Single LLM Baseline", color="salmon")
    axes[0].bar(x + width/2, c_scores, width, label="Multi-Agent Consensus", color="skyblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].set_ylabel("Rate / Score")
    axes[0].set_title("Performance & Hallucination Benchmarks")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--")
    
    # Plot 2: Latency comparison
    categories = ["Baseline", "Consensus"]
    latencies = [report["baseline"]["latency_ms"], report["consensus"]["latency_ms"]]
    axes[1].bar(categories, latencies, color=["lightcoral", "lightskyblue"], width=0.4)
    axes[1].set_ylabel("Inference Latency (ms)")
    axes[1].set_title("System Latency Comparison")
    axes[1].grid(axis="y", linestyle="--")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Consensus dashboard chart saved to: {output_path}")


def run_consensus_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTI-AGENT CLINICAL CONSENSUS PIPELINE RUNNER ")
    print("="*75)
    
    # 1. Load Data Splits
    print("[1/3] Loading patient datasets (PatientCraft)...")
    dm = MedicalDataModule(
        batch_size=4,
        train_samples=20,
        val_samples=15,
        test_samples=15,
        image_size=(384, 384)
    )
    dm.setup()
    
    # 2. Instantiate Consensus Engine & Evaluator
    print("\n[2/3] Initializing Multi-Agent Coordinator and Specialists...")
    coordinator = CoordinatorAgent()
    evaluator = ConsensusEvaluator(coordinator)
    
    # Benchmark on Validation split (In-Distribution)
    print("\nEvaluating In-Distribution Validation set...")
    val_loader = dm.val_dataloader()
    val_report = evaluator.evaluate_dataset(val_loader)
    evaluator.print_consensus_report(val_report)
    
    # Benchmark on OOD Test split (High scan noise)
    print("Evaluating OOD Test set (Severe scanner degradation)...")
    test_loader = dm.test_dataloader()
    test_report = evaluator.evaluate_dataset(test_loader)
    evaluator.print_consensus_report(test_report)
    
    # Generate dashboard plots
    generate_consensus_plots(test_report, output_path="docs/consensus_benchmark_dashboard.png")
    
    # 3. Clinical Walkthrough of a single patient case
    print("\n[3/3] Running detailed clinical walkthrough on single Pneumonia case...")
    # Create a simulated pneumonia case
    sample_patient = {
        "clinical_note": "Patient presents with cough and fever. Crackles in left lower zone. Radiograph shows lung haziness.",
        "lab_metrics": {"temperature": 102.1, "heart_rate": 102.0, "wbc_count": 14500.0, "oxygen_saturation": 92.0}
    }
    
    consensus_state = coordinator.run_consensus(sample_patient)
    res = consensus_state.coordinator_consensus
    
    print("\n" + "="*70)
    print(" CLINICAL CONSENSUS BOARD DIAGNOSTIC RECAP ")
    print("="*70)
    print(f"Patient Note:  \"{sample_patient['clinical_note']}\"")
    print(f"Patient Vitals: Temp: {sample_patient['lab_metrics']['temperature']}F, WBC: {sample_patient['lab_metrics']['wbc_count']}/uL, SpO2: {sample_patient['lab_metrics']['oxygen_saturation']}%")
    print("-"*70)
    print(f"CONSENSUS DIAGNOSIS : {res['consensus_diagnosis']}")
    print(f"ICD-10 Code         : {res['icd10_code']}")
    print(f"Agreement Rate      : {res['agreement_rate']:.2%}")
    print(f"Consensus Uncertainty: {res['consensus_uncertainty']:.2%}")
    print("-"*70)
    
    print("SPECIALIST OPINIONS & EVIDENCE:")
    for role, opinion in consensus_state.specialist_opinions.items():
        print(f" - [{role:18}]: {opinion.diagnosis} (Conf: {opinion.confidence:.2f}) | {opinion.evidence[:90]}...")
        
    print("-"*70)
    print(f"DISAGREEMENTS DETECTED ({len(res['disagreements'])}):")
    for d in res["disagreements"]:
        print(f" - Specialist '{d['specialist']}' disagreed! Suggested: {d['suggested_diagnosis']} (Conf: {d['confidence']:.2f}). Reason: {d['evidence'][:80]}...")
        
    print("-"*70)
    print("RECOMMENDED PROTOCOLS (MERGED):")
    print(f" - Tests      : {res['final_tests']}")
    print(f" - Treatments : {res['final_treatments']}")
    print("-"*70)
    print("MEDICAL RAG EVIDENCE CITATION:")
    print(f" - Guideline Source : {res['literature_citation']}")
    print(f" - Clinical Text     : \"{res['literature_guideline']}\"")
    print("="*70)
    
    print("\nConsensus Pipeline Lifecycle Execution Complete.")

if __name__ == "__main__":
    run_consensus_pipeline()
