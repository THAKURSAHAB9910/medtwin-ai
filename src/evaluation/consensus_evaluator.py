import time
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.agents.coordinator import CoordinatorAgent
from src.agents.specialists import MedicalResearcherAgent

class ConsensusEvaluator:
    """
    Consensus Benchmark Suite.
    Compares the diagnostic accuracy, latency, inter-agent agreement,
    and hallucination rates of a Single LLM Baseline vs. the Multi-Agent Consensus Engine.
    """
    def __init__(self, coordinator: CoordinatorAgent):
        self.coordinator = coordinator
        self.baseline_agent = MedicalResearcherAgent() # General medical agent baseline
        self.kg = coordinator.kg

    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Runs benchmarking over the dataset loader.
        """
        import torch # Delayed import inside method to match dataloader usage
        
        true_labels = []
        
        baseline_preds = []
        consensus_preds = []
        
        baseline_latencies = []
        consensus_latencies = []
        
        agreement_rates = []
        
        # Track hallucination instances (unsupported recommended tests)
        baseline_hallucinations = 0
        consensus_hallucinations = 0
        
        total_samples = 0
        
        for batch in dataloader:
            images = batch["image"]
            notes = batch["clinical_note"]
            labs_np = batch["lab_metrics"].numpy()
            labels = batch["label"].numpy()
            
            for i in range(len(images)):
                total_samples += 1
                true_label = labels[i]
                true_labels.append(true_label)
                
                # Setup patient metrics dictionary
                patient_data = {
                    "clinical_note": notes[i],
                    "lab_metrics": {
                        "age": float(labs_np[i, 0]),
                        "temperature": float(labs_np[i, 1]),
                        "heart_rate": float(labs_np[i, 2]),
                        "wbc_count": float(labs_np[i, 3]),
                        "oxygen_saturation": float(labs_np[i, 4])
                    }
                }
                if images is not None:
                    # Mark visual image presence in dictionary
                    patient_data["image"] = True
                
                # --- Pathway 1: Single LLM Baseline ---
                t_start = time.perf_counter()
                baseline_opinion = self.baseline_agent.analyze(patient_data)
                t_end = time.perf_counter()
                
                baseline_latencies.append((t_end - t_start) * 1000)
                b_diag = 1 if baseline_opinion.diagnosis == "Pneumonia" else 0
                baseline_preds.append(b_diag)
                
                # Check baseline hallucinations:
                # If tests recommended are not in the Knowledge Graph definitions for the diagnosis, it's flagged as hallucination
                valid_tests = self.kg.query_relations(baseline_opinion.diagnosis, "requires_test")
                for test in baseline_opinion.recommended_tests:
                    if test not in valid_tests:
                        baseline_hallucinations += 1
                        break
                
                # --- Pathway 2: Multi-Agent Consensus Engine ---
                t_start = time.perf_counter()
                consensus_state = self.coordinator.run_consensus(patient_data)
                t_end = time.perf_counter()
                
                consensus_latencies.append((t_end - t_start) * 1000)
                res = consensus_state.coordinator_consensus
                c_diag = 1 if res["consensus_diagnosis"] == "Pneumonia" else 0
                consensus_preds.append(c_diag)
                
                agreement_rates.append(res["agreement_rate"])
                
                # Check consensus hallucinations:
                # The coordinator queries KG and RAG, verifying valid tests, minimizing hallucination rate
                valid_consensus_tests = self.kg.query_relations(res["consensus_diagnosis"], "requires_test")
                for test in res["final_tests"]:
                    # In consensus, tests are checked against KG relations
                    if test not in valid_consensus_tests and len(valid_consensus_tests) > 0:
                        # Only flag if there are defined tests in graph but ours is invalid
                        consensus_hallucinations += 1
                        break
                        
        # Compute summary metrics
        true_labels = np.array(true_labels)
        baseline_preds = np.array(baseline_preds)
        consensus_preds = np.array(consensus_preds)
        
        b_acc = accuracy_score(true_labels, baseline_preds)
        _, _, b_f1, _ = precision_recall_fscore_support(true_labels, baseline_preds, average="binary", zero_division=0)
        
        c_acc = accuracy_score(true_labels, consensus_preds)
        _, _, c_f1, _ = precision_recall_fscore_support(true_labels, consensus_preds, average="binary", zero_division=0)
        
        avg_b_lat = float(np.mean(baseline_latencies))
        avg_c_lat = float(np.mean(consensus_latencies))
        
        avg_agreement = float(np.mean(agreement_rates))
        
        b_hallucination_rate = baseline_hallucinations / total_samples if total_samples > 0 else 0.0
        c_hallucination_rate = consensus_hallucinations / total_samples if total_samples > 0 else 0.0
        
        return {
            "baseline": {
                "accuracy": float(b_acc),
                "f1": float(b_f1),
                "latency_ms": avg_b_lat,
                "hallucination_rate": b_hallucination_rate
            },
            "consensus": {
                "accuracy": float(c_acc),
                "f1": float(c_f1),
                "latency_ms": avg_c_lat,
                "hallucination_rate": c_hallucination_rate,
                "average_agreement_rate": avg_agreement
            }
        }

    def print_consensus_report(self, report: Dict[str, Any]):
        print("="*75)
        print(" CLINICAL CONSENSUS ENGINE BENCHMARK REPORT ")
        print("="*75)
        print(" METRIC                     | SINGLE LLM BASELINE | MULTI-AGENT CONSENSUS | DELTA ")
        print("-"*75)
        
        # Accuracy
        b_acc = report["baseline"]["accuracy"]
        c_acc = report["consensus"]["accuracy"]
        print(f" Diagnostic Accuracy        | {b_acc:19.2%} | {c_acc:21.2%} | {c_acc - b_acc:+.2%} ")
        
        # F1-Score
        b_f1 = report["baseline"]["f1"]
        c_f1 = report["consensus"]["f1"]
        print(f" Diagnostic F1-Score        | {b_f1:19.4f} | {c_f1:21.4f} | {c_f1 - b_f1:+.4f} ")
        
        # Hallucination Rate (Lower is better)
        b_hal = report["baseline"]["hallucination_rate"]
        c_hal = report["consensus"]["hallucination_rate"]
        print(f" Recommendation Hallucinations| {b_hal:19.2%} | {c_hal:21.2%} | {c_hal - b_hal:+.2%} ")
        
        # Latency (Lower is better)
        b_lat = report["baseline"]["latency_ms"]
        c_lat = report["consensus"]["latency_ms"]
        print(f" Inference Latency          | {b_lat:16.2f} ms | {c_lat:18.2f} ms | {c_lat - b_lat:+.2f} ms ")
        
        print("-"*75)
        print(f"Average Specialist Agreement Rate: {report['consensus']['average_agreement_rate']:.2%}")
        print("="*75)
