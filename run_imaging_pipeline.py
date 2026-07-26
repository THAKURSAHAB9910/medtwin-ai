import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.imaging.pipeline import MedicalImagingPipeline
from src.imaging.benchmarker import ImagingArchitectureBenchmarker

def generate_imaging_dashboard(benchmarks: dict, output_path: str = "docs/imaging_research_benchmarks.png"):
    """
    Plots multi-architecture vision performance (Accuracy/IoU vs Latency/Memory).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = list(benchmarks.keys())
    latencies = [benchmarks[m]["latency_ms"] for m in models]
    memories = [benchmarks[m]["peak_vram_mb"] for m in models]
    ious = [benchmarks[m]["segmentation_iou"] * 100 for m in models]
    accs = [benchmarks[m]["classification_accuracy"] * 100 for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Latency & VRAM Footprint
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, latencies, width, label='Latency (ms)', color='orchid')
    rects2 = axes[0].bar(x + width/2, [m/10 for m in memories], width, label='VRAM VRAM (x100 MB)', color='cornflowerblue')
    
    axes[0].set_ylabel('Resource Profile Values')
    axes[0].set_title('Vision Model Resource Footprint Profile')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--')
    
    # Plot 2: Performance Metrics (Accuracy & IoU)
    axes[1].bar(x - width/2, accs, width, label='Classification Acc (%)', color='lightgreen')
    axes[1].bar(x + width/2, ious, width, label='Segmentation IoU (%)', color='orange')
    
    axes[1].set_ylabel('Percentage (%)')
    axes[1].set_title('Task Quality Benchmark Comparisons')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Imaging research benchmark dashboard saved to: {output_path}")


def run_imaging_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTI-MODALITY IMAGING RESEARCH PIPELINE ")
    print("="*75)
    
    # 1. Initialize Pipeline
    print("[1/4] Initializing vision pipeline router...")
    pipeline = MedicalImagingPipeline()
    benchmarker = ImagingArchitectureBenchmarker()
    
    # 2. Process all 9 modalities
    print("[2/4] Testing classification & segmentations over 9 modalities...")
    mock_img = Image.new("RGB", (256, 256), color="grey")
    
    modalities = [
        "x-ray", "mri", "ct", "retinal", "skin lesion", 
        "histopathology", "microscopy", "ultrasound", "wound"
    ]
    
    for mod in modalities:
        print(f"\n --- PROCESSING MODALITY: {mod.upper()} ---")
        c_res = pipeline.classify(mock_img, mod)
        s_res = pipeline.segment(mock_img, mod)
        d_res = pipeline.detect(mock_img, mod)
        l_res = pipeline.localize(mock_img, mod)
        v_res = pipeline.estimate_severity(mock_img, mod)
        
        print(f"  * Classification:  {c_res['diagnosis']} (Conf: {c_res['confidence']:.2%})")
        print(f"  * Segmented Area:  {s_res['segmented_area_percent']}% of scan dimensions")
        print(f"  * Anomaly Bounding Box: {d_res['bounding_box']}")
        print(f"  * Lesion Centroid Coordinate: {l_res['centroid']}")
        print(f"  * Severity Estimate: {v_res['severity_level']} (Score: {v_res['severity_score']:.2f})")
        
    # 3. Trigger Architectural Comparison
    print("\n[3/4] Running multi-architecture benchmarks...")
    benchmarks = benchmarker.run_comparative_benchmark()
    
    print("\n" + "="*85)
    print(" MEDICAL IMAGING MULTI-ARCHITECTURE BENCHMARKS ")
    print("="*85)
    print(" ARCHITECTURE | CLASSIFY ACC | SEGMENT IoU | DETECTION mAP | LATENCY | PEAK VRAM ")
    print("-"*85)
    for model, metrics in benchmarks.items():
        print(f" {model:12} | {metrics['classification_accuracy']:12.1%} | {metrics['segmentation_iou']:11.1%} | {metrics['detection_map']:13.1%} | {metrics['latency_ms']:5.1f} ms | {metrics['peak_vram_mb']:7.1f} MB ")
    print("="*85)
    
    # 4. Generate Visualization Charts
    print("\n[4/4] Plotting visual vision resource comparisons...")
    generate_imaging_dashboard(benchmarks, "docs/imaging_research_benchmarks.png")
    
    print("\nMedical Image Research pipeline execution complete.")

if __name__ == "__main__":
    run_imaging_pipeline()
