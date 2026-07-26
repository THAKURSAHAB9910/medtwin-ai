import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from src.data.generator import PatientCraftEngine
from src.finetuning.dataset import ClinicalSFTDataset, MockTokenizer, create_sft_dataloader
from src.finetuning.trainer import ClinicalPEFTTrainer
from src.finetuning.hpo import ClinicalHPOSearch
from src.finetuning.benchmark import AdapterPerformanceBenchmarker

class MockModel(nn.Module):
    """
    Mock PyTorch neural module matching standard VLM forward pass architectures.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(224 * 224 * 3, 10)
        
    def forward(self, x):
        # Flatten image inputs and project to logit classes
        x_flat = x.view(x.size(0), -1)
        return self.proj(x_flat)

def generate_finetune_comparison_chart(results: dict, output_path: str = "docs/finetune_ablation_comparison.png"):
    """
    Generates comparative plots comparing adapter diagnostics, VRAM requirements, and training times.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    configs = list(results.keys())
    accuracies = [results[c]["accuracy"] * 100 for c in configs]
    f1s = [results[c]["f1"] for c in configs]
    vram = [results[c]["gpu_vram_mb"] for c in configs]
    train_time = [results[c]["training_time_sec"] for c in configs]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy & F1
    x = np.arange(len(configs))
    width = 0.35
    axes[0].bar(x - width/2, accuracies, width, label="Accuracy (%)", color="mediumslateblue")
    axes[0].bar(x + width/2, [f * 100 for f in f1s], width, label="F1-Score (x100)", color="aquamarine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs)
    axes[0].set_ylabel("Performance Metric")
    axes[0].set_title("PEFT Adapter Validation Performance")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--")
    
    # Plot 2: Peak VRAM vs Training Time
    axes[1].set_xlabel('Adapter Configuration')
    axes[1].set_ylabel('Peak GPU VRAM (MB)', color='cornflowerblue')
    axes[1].bar(x - width/2, vram, width, color='cornflowerblue', alpha=0.8, label="VRAM (MB)")
    axes[1].tick_params(axis='y', labelcolor='cornflowerblue')
    
    ax2 = axes[1].twinx()
    ax2.set_ylabel('Training Time (seconds)', color='lightcoral')
    ax2.bar(x + width/2, train_time, width, color='lightcoral', alpha=0.8, label="Training Time (s)")
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs)
    axes[1].set_title("GPU Resource Allocations vs. Training Speed")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PEFT comparative ablation chart saved to: {output_path}")


def run_finetuning_pipeline():
    print("="*75)
    print(" MEDTWIN AI: CLINICAL ADAPTER PEFT FINE-TUNING PIPELINE ")
    print("="*75)
    
    # 1. Synthesize patient SFT datasets
    print("[1/5] Synthesizing SFT cohorts...")
    engine = PatientCraftEngine(seed=45)
    train_records = [engine.generate_patient_record(has_pneumonia=(i % 2 == 0)) for i in range(40)]
    val_records = [engine.generate_patient_record(has_pneumonia=(i % 2 != 0)) for i in range(10)]
    
    tokenizer = MockTokenizer()
    train_loader = create_sft_dataloader(train_records, tokenizer, batch_size=4)
    val_loader = create_sft_dataloader(val_records, tokenizer, batch_size=4)
    
    model = MockModel()
    
    # 2. Run Hyperparameter Optimization (HPO) Sweep
    print("\n[2/5] Starting Hyperparameter Optimization (HPO) Sweep...")
    hpo_search = ClinicalHPOSearch(learning_rates=[1e-4, 2e-4], lora_ranks=[8, 16])
    hpo_results = hpo_search.run_grid_search(model, train_loader, val_loader)
    best_cfg = hpo_results["best_config"]
    
    # 3. Run Comparative Ablation Trials
    print("\n[3/5] Launching comparative training trials...")
    benchmarker = AdapterPerformanceBenchmarker()
    results = benchmarker.run_adapter_comparison(model, train_loader, val_loader)
    
    # Print formatted ablation matrix
    benchmarker.print_ablation_matrix(results)
    
    # 4. Save and Export Ablation Visualization
    print("\n[4/5] Plotting resource benchmarks...")
    generate_finetune_comparison_chart(results, output_path="docs/finetune_ablation_comparison.png")
    
    # 5. Checkpoint Save/Resume Verification
    print("\n[5/5] Verifying checkpoint save and resume states...")
    test_trainer = ClinicalPEFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="checkpoints/verification",
        learning_rate=best_cfg["learning_rate"],
        lora_r=best_cfg["lora_r"]
    )
    
    # Save a checkpoint manually
    test_trainer.save_checkpoint(step=20)
    
    # Instantiate a new trainer and load the saved checkpoint
    new_trainer = ClinicalPEFTTrainer(
        model=MockModel(),
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir="checkpoints/verification"
    )
    success = new_trainer.resume_training("checkpoints/verification/checkpoint-step-20")
    if success:
        print("Checkpoint resume verification: PASSED")
    else:
        print("Checkpoint resume verification: FAILED")
        
    print("\nClinical PEFT fine-tuning pipeline execution complete.")

if __name__ == "__main__":
    run_finetuning_pipeline()
