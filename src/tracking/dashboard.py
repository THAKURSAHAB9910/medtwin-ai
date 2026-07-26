import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

class ExperimentDashboardGenerator:
    """
    Compiles hyperparameter runs and formats visual dashboards
    comparing resource trade-offs and validation accuracies.
    """
    def generate_tracking_dashboard(self, runs: List[Dict[str, Any]], output_path: str = "docs/experiment_tracking_dashboard.png"):
        """
        Plots a 2x2 grid tracking F1 accuracy, GPU memory loads, training times,
        and hallucination rates across registered model iterations.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        run_names = [r["run_name"] for r in runs]
        lora_ranks = [r["parameters"].get("lora_rank", 0) for r in runs]
        lrs = [r["parameters"].get("learning_rate", 0) for r in runs]
        
        vram_usages = [r["metrics"].get("peak_vram_gb", 0) for r in runs]
        f1_scores = [r["metrics"].get("validation_f1", 0) * 100 for r in runs]
        latencies = [r["metrics"].get("latency_ms", 0) for r in runs]
        hallucinations = [r["metrics"].get("hallucination_rate", 0) * 100 for r in runs]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        x = np.arange(len(run_names))
        width = 0.35
        
        # 1. Validation F1 Accuracy vs. LoRA Rank
        axes[0, 0].bar(x, f1_scores, color="mediumseagreen", width=0.4)
        axes[0, 0].set_ylabel("F1 Score (%)")
        axes[0, 0].set_title("Validation Accuracy vs. LoRA Rank configurations")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels([f"Rank={r}" for r in lora_ranks])
        axes[0, 0].grid(axis="y", linestyle="--")
        
        # 2. Peak GPU memory load (GB)
        axes[0, 1].bar(x, vram_usages, color="coral", width=0.4)
        axes[0, 1].set_ylabel("Peak VRAM (GB)")
        axes[0, 1].set_title("GPU VRAM load per run config")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(run_names)
        axes[0, 1].grid(axis="y", linestyle="--")
        
        # 3. Model Latency (ms)
        axes[1, 0].plot(x, latencies, marker="o", color="royalblue", linewidth=2)
        axes[1, 0].set_ylabel("Inference Latency (ms)")
        axes[1, 0].set_title("Model Inference Latency timeline")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(run_names)
        axes[1, 0].grid(True, linestyle="--")
        
        # 4. RAG Hallucination rate
        axes[1, 1].plot(x, hallucinations, marker="s", color="crimson", linewidth=2)
        axes[1, 1].set_ylabel("Hallucination Rate (%)")
        axes[1, 1].set_title("Hallucination mitigation curve")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(run_names)
        axes[1, 1].grid(True, linestyle="--")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Hyperparameter experiment tracking dashboard saved to: {output_path}")
