import os
import matplotlib.pyplot as plt
import numpy as np

from src.data.generator import PatientCraftEngine
from src.agents.coordinator import CoordinatorAgent

def generate_consensus_chart(results: dict, output_path: str = "docs/agents_consensus_comparison.png"):
    """
    Generates diagnostic charts plotting consensus margins and individual specialist confidences.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data
    agent_names = list(results["specialist_opinions"].keys())
    confidences = [float(op.confidence) * 100 for op in results["specialist_opinions"].values()]
    diagnoses = [op.diagnosis for op in results["specialist_opinions"].values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Agent specific confidences
    axes[0].barh(agent_names, confidences, color='skyblue', edgecolor='navy')
    axes[0].set_xlabel('Confidence (%)')
    axes[0].set_title('Specialist Agent Diagnostic Confidence Ratings')
    axes[0].grid(axis='x', linestyle='--')
    
    # Plot 2: Diagnosis Vote Distribution
    unique_diags, counts = np.unique(diagnoses, return_counts=True)
    axes[1].pie(counts, labels=unique_diags, autopct='%1.1f%%', colors=['lightgreen', 'salmon', 'gold', 'lightblue'], startangle=140)
    axes[1].set_title('Multi-Agent Diagnosis Vote Distribution')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Multi-Agent consensus evaluation dashboard saved to: {output_path}")


def run_agents_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTI-AGENT CLINICAL REASONING CONSENSUS BOARD ")
    print("="*75)
    
    # 1. Synthesize Patient Data
    print("[1/4] Synthesizing complex multi-system case...")
    engine = PatientCraftEngine(seed=45)
    patient = engine.generate_patient_record(has_pneumonia=True)
    
    # Inject drug safety triggers
    patient["clinical_note"] = patient["clinical_note"] + " Scheduled for scan with Contrast Dye. Active Metformin use."
    
    # 2. Run Coordinator Agent Consensus
    print("[2/4] Triggering Specialist Agent evaluations and consensus routing...")
    coordinator = CoordinatorAgent()
    state = coordinator.run_consensus(patient)
    res = state.coordinator_consensus
    
    # 3. Analyze Consensus and Disagreements
    print("\n[3/4] Multi-Agent Clinical Board Consensus:")
    print("="*65)
    print(f" CONSENSUS DIAGNOSIS: {res['consensus_diagnosis']} ({res['icd10_code']})")
    print(f" AGREEMENT RATE:      {res['agreement_rate']:.2%}")
    print(f" UNCERTAINTY RATING:  {res['consensus_uncertainty']:.2%}")
    print(f" LITERATURE CITATION: {res['literature_citation']}")
    print("="*65)
    
    print("\n[4/4] Disagreement and Differential Analysis:")
    if len(res["disagreements"]) == 0:
        print("  All specialist agents achieved perfect consensus alignment.")
    else:
        for dis in res["disagreements"]:
            print(f"  * {dis['specialist']} disagreed:")
            print(f"    - Diagnosed:  {dis['suggested_diagnosis']} (Confidence: {dis['confidence']:.2%})")
            print(f"    - Evidence:   {dis['evidence']}")
            
    # Compile results for charting
    chart_data = {
        "specialist_opinions": state.specialist_opinions
    }
    generate_consensus_chart(chart_data, output_path="docs/agents_consensus_comparison.png")
    
    print("\nMulti-Agent Clinical Reasoning pipeline execution complete.")

if __name__ == "__main__":
    run_agents_pipeline()
