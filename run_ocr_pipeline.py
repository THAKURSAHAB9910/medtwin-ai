import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.nlp.entity_extractor import ClinicalNER
from src.ocr.pipeline import MedicalOCRPipeline
from src.ocr.benchmarker import OCRArchitectureBenchmarker

def generate_ocr_dashboard(benchmarks: dict, output_path: str = "docs/ocr_research_benchmarks.png"):
    """
    Plots multi-architecture OCR performance (Accuracy vs Latency/Robustness).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = list(benchmarks.keys())
    latencies = [benchmarks[m]["latency_ms"] for m in models]
    robustness = [benchmarks[m]["robustness_score"] * 10 for m in models]
    char_accs = [benchmarks[m]["character_accuracy"] * 100 for m in models]
    term_accs = [benchmarks[m]["medical_term_accuracy"] * 100 for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Character & Terminology Accuracies
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, char_accs, width, label='Character Acc (%)', color='skyblue')
    axes[0].bar(x + width/2, term_accs, width, label='Medical Term Acc (%)', color='lightgreen')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('OCR Text Recognition Quality Benchmarks')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--')
    
    # Plot 2: Latency & Robustness
    axes[1].bar(x - width/2, latencies, width, label='Latency (ms)', color='lightcoral')
    axes[1].bar(x + width/2, robustness, width, label='Robustness (x10 %)', color='gold')
    axes[1].set_ylabel('Scores / Speed')
    axes[1].set_title('Inference Latency & Noise Robustness Ratios')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"OCR research benchmark dashboard saved to: {output_path}")


def run_ocr_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTI-ENGINE MEDICAL OCR RESEARCH PIPELINE ")
    print("="*75)
    
    # 1. Initialize Pipeline
    print("[1/4] Initializing OCR pipeline router and entity extractors...")
    ner = ClinicalNER()
    pipeline = MedicalOCRPipeline(ner)
    benchmarker = OCRArchitectureBenchmarker()
    
    # 2. Process all 7 document types
    print("[2/4] Executing OCR text & entity extraction over 7 document templates...")
    mock_img = Image.new("RGB", (300, 150), color="white")
    
    doc_types = [
        "prescription", "lab report", "discharge summary", 
        "insurance document", "referral letter", 
        "handwritten prescription", "medical certificate"
    ]
    
    for dt in doc_types:
        print(f"\n --- PROCESSING DOCUMENT: {dt.upper()} ---")
        res = pipeline.extract_text_and_entities(mock_img, dt)
        print(f"  * Recognized Text: \"{res['ocr_text']}\"")
        print(f"  * Extracted Meds:  {res['extracted_medications']}")
        print(f"  * Extracted Dosages: {res['extracted_dosages']}")
        print(f"  * Extracted Diseases: {res['extracted_diseases']}")
        
    # 3. Trigger OCR Comparison
    print("\n[3/4] Running multi-architecture OCR benchmarks...")
    benchmarks = benchmarker.run_comparative_benchmark()
    
    print("\n" + "="*95)
    print(" MEDICAL DOCUMENT OCR ENGINE BENCHMARKS ")
    print("="*95)
    print(" ENGINE       | CHAR ACC | TERM ACC | DRUG RECALL | DOSAGE RECALL | LATENCY  | ROBUSTNESS ")
    print("-"*95)
    for model, metrics in benchmarks.items():
        print(f" {model:12} | {metrics['character_accuracy']:8.1%} | {metrics['medical_term_accuracy']:8.1%} | {metrics['drug_extraction_recall']:11.1%} | {metrics['dosage_extraction_recall']:13.1%} | {metrics['latency_ms']:5.1f} ms | {metrics['robustness_score']:7.1f}/10 ")
    print("="*95)
    
    # 4. Generate Visualization Charts
    print("\n[4/4] Plotting visual OCR comparisons...")
    generate_ocr_dashboard(benchmarks, "docs/ocr_research_benchmarks.png")
    
    print("\nMedical OCR and Document Intelligence pipeline execution complete.")

if __name__ == "__main__":
    run_ocr_pipeline()
