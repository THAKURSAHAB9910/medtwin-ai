import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any

class ExperimentReporter:
    """
    Compiles experimental runs into publication-ready scientific reports
    and formats visual evaluation dashboards.
    """
    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def generate_markdown_report(self, output_path: str = "docs/experimental_report.md") -> str:
        """
        Generates a structured scientific report markdown file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        md = []
        md.append("# MedTwin AI: Clinical Research Experiments Report\n")
        md.append("This document compiles scientific validation experiments and profiles key clinical hypotheses.\n")
        md.append("## Table of Contents")
        md.append("1. [Multimodal Diagnostics Evaluation](#1-multimodal-diagnostics-evaluation)")
        md.append("2. [LoRA / QLoRA Parameter Efficiency Study](#2-lora--qlora-parameter-efficiency-study)")
        md.append("3. [Clinical RAG Hallucination Audit](#3-clinical-rag-hallucination-audit)")
        md.append("4. [Synthetic Dataset Robustness Assessment](#4-synthetic-dataset-robustness-assessment)\n")
        md.append("---\n")
        
        # Section 1: Multimodal
        r_mm = self.results["multimodal_reasoning"]
        md.append("## 1. Multimodal Diagnostics Evaluation")
        md.append(f"**Research Question**: {r_mm['research_question']}")
        md.append(f"**Hypothesis**: {r_mm['hypothesis']}")
        md.append(f"**Baseline Configuration**: {r_mm['baseline']}")
        md.append(f"**Improved Model**: {r_mm['improved_model']}")
        md.append(f"**Dataset**: {r_mm['dataset']}\n")
        md.append("### Quantitative Metrics:")
        md.append(f"* Text-Only Precision: {r_mm['metrics']['text_baseline_precision']:.1%}")
        md.append(f"* Image-Only Precision: {r_mm['metrics']['image_baseline_precision']:.1%}")
        md.append(f"* Multimodal Fuser Precision: **{r_mm['metrics']['multimodal_fuser_precision']:.1%}**")
        md.append(f"* F1-Score (Multimodal): {r_mm['metrics']['f1_score_multimodal']:.2f}")
        md.append(f"* Inference Latency: {r_mm['metrics']['latency_ms']} ms\n")
        md.append(f"**Results**: {r_mm['results']}")
        md.append(f"**Discussion**: {r_mm['discussion']}")
        md.append(f"**Conclusion**: *{r_mm['conclusion']}*\n")
        md.append("---\n")
        
        # Section 2: LoRA
        r_lora = self.results["lora_efficiency"]
        md.append("## 2. LoRA / QLoRA Parameter Efficiency Study")
        md.append(f"**Research Question**: {r_lora['research_question']}")
        md.append(f"**Hypothesis**: {r_lora['hypothesis']}")
        md.append(f"**Baseline Configuration**: {r_lora['baseline']}")
        md.append(f"**Improved Model**: {r_lora['improved_model']}")
        md.append(f"**Dataset**: {r_lora['dataset']}\n")
        md.append("### Quantitative Metrics:")
        md.append(f"* Full Fine-Tuning Accuracy: {r_lora['metrics']['full_tuning_accuracy']:.1%}")
        md.append(f"* LoRA Adapter Accuracy: {r_lora['metrics']['lora_tuning_accuracy']:.1%}")
        md.append(f"* QLoRA 4-bit Accuracy: {r_lora['metrics']['qlora_tuning_accuracy']:.1%}")
        md.append(f"* Full VRAM Footprint: {r_lora['metrics']['full_vram_gb']} GB")
        md.append(f"* LoRA VRAM Footprint: **{r_lora['metrics']['lora_vram_gb']} GB**")
        md.append(f"* QLoRA VRAM Footprint: **{r_lora['metrics']['qlora_vram_gb']} GB**\n")
        md.append(f"**Results**: {r_lora['results']}")
        md.append(f"**Discussion**: {r_lora['discussion']}")
        md.append(f"**Conclusion**: *{r_lora['conclusion']}*\n")
        md.append("---\n")
        
        # Section 3: RAG
        r_rag = self.results["clinical_rag_hallucination"]
        md.append("## 3. Clinical RAG Hallucination Audit")
        md.append(f"**Research Question**: {r_rag['research_question']}")
        md.append(f"**Hypothesis**: {r_rag['hypothesis']}")
        md.append(f"**Baseline Configuration**: {r_rag['baseline']}")
        md.append(f"**Improved Model**: {r_rag['improved_model']}")
        md.append(f"**Dataset**: {r_rag['dataset']}\n")
        md.append("### Quantitative Metrics:")
        md.append(f"* Zero-Shot Hallucination Rate: {r_rag['metrics']['zero_shot_hallucination_rate']:.1%}")
        md.append(f"* RAG-Augmented Hallucination Rate: **{r_rag['metrics']['rag_hallucination_rate']:.1%}**")
        md.append(f"* Retrieval Reciprocal Rank: {r_rag['metrics']['retrieval_reciprocal_rank']:.2f}")
        md.append(f"* QA Faithfulness Score: {r_rag['metrics']['qa_faithfulness_score']:.2f}\n")
        md.append(f"**Results**: {r_rag['results']}")
        md.append(f"**Discussion**: {r_rag['discussion']}")
        md.append(f"**Conclusion**: *{r_rag['conclusion']}*\n")
        md.append("---\n")
        
        # Section 4: Synthetic
        r_synth = self.results["synthetic_robustness"]
        md.append("## 4. Synthetic Dataset Robustness Assessment")
        md.append(f"**Research Question**: {r_synth['research_question']}")
        md.append(f"**Hypothesis**: {r_synth['hypothesis']}")
        md.append(f"**Baseline Configuration**: {r_synth['baseline']}")
        md.append(f"**Improved Model**: {r_synth['improved_model']}")
        md.append(f"**Dataset**: {r_synth['dataset']}\n")
        md.append("### Quantitative Metrics:")
        md.append(f"* Clean-Trained model on Noisy Test set (F1): {r_synth['metrics']['clean_trained_noisy_f1']:.2f}")
        md.append(f"* Augmented-Trained model on Noisy Test set (F1): **{r_synth['metrics']['augmented_trained_noisy_f1']:.2f}**\n")
        md.append(f"**Results**: {r_synth['results']}")
        md.append(f"**Discussion**: {r_synth['discussion']}")
        md.append(f"**Conclusion**: *{r_synth['conclusion']}*\n")
        
        content = "\n".join(md)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        print(f"Scientific experiments report saved to: {output_path}")
        return content

    def generate_comparison_charts(self, output_path: str = "docs/experimental_results.png"):
        """
        Generates a 2x2 grid visualizing metrics across all 4 experiments.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Multimodal
        r_mm = self.results["multimodal_reasoning"]
        mm_labels = ["Image Only", "Text Only", "Multimodal"]
        mm_vals = [
            r_mm["metrics"]["image_baseline_precision"] * 100,
            r_mm["metrics"]["text_baseline_precision"] * 100,
            r_mm["metrics"]["multimodal_fuser_precision"] * 100
        ]
        axes[0, 0].bar(mm_labels, mm_vals, color=["coral", "cornflowerblue", "mediumseagreen"])
        axes[0, 0].set_ylabel("Precision (%)")
        axes[0, 0].set_title("Exp 1: Multimodal Fusion Accuracy Gain")
        axes[0, 0].grid(axis="y", linestyle="--")
        
        # Panel 2: LoRA Memory
        r_lora = self.results["lora_efficiency"]
        lora_labels = ["Full Fine-Tuning", "LoRA PEFT", "QLoRA 4-bit"]
        lora_vals = [
            r_lora["metrics"]["full_vram_gb"],
            r_lora["metrics"]["lora_vram_gb"],
            r_lora["metrics"]["qlora_vram_gb"]
        ]
        axes[0, 1].bar(lora_labels, lora_vals, color=["red", "orange", "gold"])
        axes[0, 1].set_ylabel("Peak VRAM (GB)")
        axes[0, 1].set_title("Exp 2: GPU VRAM Memory Footprint Reduction")
        axes[0, 1].grid(axis="y", linestyle="--")
        
        # Panel 3: RAG Hallucination Rate
        r_rag = self.results["clinical_rag_hallucination"]
        rag_labels = ["Zero-Shot Summary", "Clinical RAG"]
        rag_vals = [
            r_rag["metrics"]["zero_shot_hallucination_rate"] * 100,
            r_rag["metrics"]["rag_hallucination_rate"] * 100
        ]
        axes[1, 0].bar(rag_labels, rag_vals, color=["crimson", "teal"])
        axes[1, 0].set_ylabel("Hallucination Rate (%)")
        axes[1, 0].set_title("Exp 3: Hallucination Mitigation Index")
        axes[1, 0].grid(axis="y", linestyle="--")
        
        # Panel 4: Synthetic Generalization
        r_synth = self.results["synthetic_robustness"]
        synth_labels = ["Clean-Trained Model", "Augmented-Trained Model"]
        synth_vals = [
            r_synth["metrics"]["clean_trained_noisy_f1"],
            r_synth["metrics"]["augmented_trained_noisy_f1"]
        ]
        axes[1, 1].bar(synth_labels, synth_vals, color=["grey", "purple"])
        axes[1, 1].set_ylabel("Noisy Test F1-Score")
        axes[1, 1].set_title("Exp 4: Noise Robustness & Generalization")
        axes[1, 1].grid(axis="y", linestyle="--")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Experimental comparison charts saved to: {output_path}")
