import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.vlm.unified_api import MedicalVLMRegistry
from src.vlm.explain import VLMExplainer
from src.vlm.benchmark import VLMPerformanceBenchmarker

def generate_vlm_comparison_chart(results: dict, output_path: str = "docs/vlm_benchmark_comparison.png"):
    """
    Generates comparative plots comparing VLM performance, latency, and memory footprints.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = list(results.keys())
    diag_acc = [results[m]["diagnosis_accuracy"] * 100 for m in models]
    ocr_acc = [(1.0 - results[m]["ocr_cer"]) * 100 for m in models]
    latencies = [results[m]["latency_ms"] for m in models]
    vram = [results[m]["vram_mb"] for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy & OCR comparison
    x = np.arange(len(models))
    width = 0.35
    axes[0].bar(x - width/2, diag_acc, width, label="Diagnosis Accuracy", color="mediumslateblue")
    axes[0].bar(x + width/2, ocr_acc, width, label="OCR Text Accuracy", color="aquamarine")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("VLM Clinical Accuracy Benchmarks")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--")
    
    # Plot 2: Latency vs VRAM (dual axis)
    color = 'lightcoral'
    axes[1].set_xlabel('VLM Model')
    axes[1].set_ylabel('Inference Latency (ms)', color=color)
    bars = axes[1].bar(x, latencies, color=color, alpha=0.6, width=0.4, label="Latency")
    axes[1].tick_params(axis='y', labelcolor=color)
    
    ax2 = axes[1].twinx()
    color2 = 'cornflowerblue'
    ax2.set_ylabel('GPU VRAM footprint (MB)', color=color2)
    line = ax2.plot(x, vram, color=color2, marker='o', linewidth=2.5, label="VRAM (MB)")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].set_title("Latency vs. VRAM Memory Requirements")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"VLM comparison dashboard chart saved to: {output_path}")


def run_vlm_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTIMODAL MEDICAL VLM BENCHMARK PIPELINE ")
    print("="*75)
    
    # 1. Initialize Registry and Benchmarker
    print("[1/3] Loading VLM registry components...")
    registry = MedicalVLMRegistry(use_mock=True)
    benchmarker = VLMPerformanceBenchmarker(registry)
    
    # 2. Run Benchmarks
    print("[2/3] Evaluating models (Qwen2-VL, BioMedCLIP, Florence-2, LayoutLMv3)...")
    # Ingests a dummy loader list
    results = benchmarker.benchmark_all(dataloader=None)
    benchmarker.print_benchmark_matrix(results)
    
    # Generate visualization chart
    generate_vlm_comparison_chart(results, output_path="docs/vlm_benchmark_comparison.png")
    
    # 3. Grounded Visual Explainability Walkthrough (Florence-2 & VLMExplainer)
    print("\n[3/3] Running Florence-2 Visual Anomaly Grounding simulation...")
    explainer = VLMExplainer(output_dir="docs")
    
    # Create mock chest radiograph
    mock_scan = Image.new("RGB", (512, 512), color="lightgray")
    
    # Run Florence-2 inference for grounding
    registry.switch_model("florence-2")
    florence_res = registry.infer(mock_scan, "ground lung consolidation opacity", task="grounding")
    
    bboxes = florence_res["outputs"]["grounded_box"]
    diag = florence_res["outputs"]["diagnosis"]
    
    # Generate bounded overlay image
    grounded_image_path = explainer.draw_grounded_boxes(
        image=mock_scan,
        bboxes=bboxes,
        label="Consolidation Area",
        save_name="vlm_grounded_output.png"
    )
    print(f"Grounded radiographic scan exported successfully: {grounded_image_path}")
    
    # Generate Qwen2-VL report trace explanation
    registry.switch_model("qwen2-vl")
    qwen_res = registry.infer(mock_scan, "explain lung consolodations and wbc counts", task="reasoning")
    trace = explainer.generate_clinical_trace(qwen_res)
    print("\n" + trace)
    
    print("\nMultimodal VLM pipeline execution complete.")

if __name__ == "__main__":
    run_vlm_pipeline()
