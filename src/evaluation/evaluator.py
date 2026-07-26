import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Any, List, Tuple

from src.training.trainer import MedTwinAIModule
from src.training.self_healing import ClinicalSelfHealingEngine

class MedTwinAIEvaluator:
    """
    Evaluator class for MedTwin AI.
    Calculates diagnostic performance, calibration metrics, OOD robustness, 
    and measures the improvements obtained by self-healing physiological reasoning.
    """
    def __init__(self, model: MedTwinAIModule, self_healing_engine: ClinicalSelfHealingEngine):
        self.model = model
        self.self_healing = self_healing_engine
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        """
        Runs complete evaluation over the provided dataloader.
        """
        raw_probs = []
        healed_probs = []
        labels = []
        
        raw_preds = []
        healed_preds = []
        
        latencies = []
        
        action_counts = {"none": 0, "healed_normal_verification": 0, "healed_infection_verification": 0, "routed_to_physician": 0}
        
        for batch in dataloader:
            images = batch["image"].to(self.model.device)
            notes = batch["clinical_note"]
            labs = batch["lab_metrics"].to(self.model.device)
            batch_labels = batch["label"].cpu().numpy()
            
            # Reconstruct lab metrics dictionary for each sample in the batch to pass to the healer
            # Vector shape: [B, 5] -> [age, temperature, heart_rate, wbc_count, oxygen_saturation]
            labs_np = batch["lab_metrics"].numpy()
            
            start_time = time.perf_counter()
            outputs = self.model(images, notes, labs)
            
            fused_logits = outputs["fused_logit"]
            probs = self.model.calibrator.get_calibrated_probability(fused_logits).squeeze(-1).cpu().numpy()
            lesion_maps = torch.sigmoid(outputs["lesion_map"]).cpu().numpy()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000 / len(images))
            
            for i in range(len(images)):
                prob = float(probs[i])
                lesion_map = lesion_maps[i]
                
                # Fetch vitals
                sample_labs = {
                    "age": float(labs_np[i, 0]),
                    "temperature": float(labs_np[i, 1]),
                    "heart_rate": float(labs_np[i, 2]),
                    "wbc_count": float(labs_np[i, 3]),
                    "oxygen_saturation": float(labs_np[i, 4])
                }
                
                # Apply Clinical Self-Healing
                healed = self.self_healing.verify_and_heal(prob, lesion_map, sample_labs)
                
                raw_probs.append(prob)
                raw_preds.append(int(prob >= 0.5))
                
                healed_probs.append(healed["healed_prob"])
                healed_preds.append(healed["healed_label"])
                labels.append(batch_labels[i])
                
                act = healed["action_taken"]
                action_counts[act] = action_counts.get(act, 0) + 1
                
        labels = np.array(labels)
        raw_probs = np.array(raw_probs)
        healed_probs = np.array(healed_probs)
        raw_preds = np.array(raw_preds)
        healed_preds = np.array(healed_preds)
        
        # Base classification metrics
        raw_acc = accuracy_score(labels, raw_preds)
        raw_prec, raw_rec, raw_f1, _ = precision_recall_fscore_support(labels, raw_preds, average="binary", zero_division=0)
        try:
            raw_auroc = roc_auc_score(labels, raw_probs)
        except ValueError:
            raw_auroc = 0.5
        raw_ece = self.model.calibrator.compute_ece(raw_probs, labels)
        
        # Healed classification metrics
        healed_acc = accuracy_score(labels, healed_preds)
        healed_prec, healed_rec, healed_f1, _ = precision_recall_fscore_support(labels, healed_preds, average="binary", zero_division=0)
        try:
            healed_auroc = roc_auc_score(labels, healed_probs)
        except ValueError:
            healed_auroc = 0.5
        healed_ece = self.model.calibrator.compute_ece(healed_probs, labels)
        
        # Conformal prediction checks
        conformal_sets = self.model.calibrator.predict_conformal_set(healed_probs)
        conformal_coverage = np.mean([labels[idx] in conformal_sets[idx] for idx in range(len(labels))])
        conformal_ambiguity = np.mean([len(c_set) > 1 for c_set in conformal_sets])
        
        avg_latency = float(np.mean(latencies))
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
        
        return {
            "raw": {
                "accuracy": float(raw_acc),
                "precision": float(raw_prec),
                "recall": float(raw_rec),
                "f1": float(raw_f1),
                "auroc": float(raw_auroc),
                "ece": float(raw_ece)
            },
            "healed": {
                "accuracy": float(healed_acc),
                "precision": float(healed_prec),
                "recall": float(healed_rec),
                "f1": float(healed_f1),
                "auroc": float(healed_auroc),
                "ece": float(healed_ece)
            },
            "self_healing_telemetry": {
                "actions_triggered": action_counts,
                "conformal_empirical_coverage": float(conformal_coverage),
                "conformal_ambiguity_rate": float(conformal_ambiguity)
            },
            "system_telemetry": {
                "avg_latency_ms": avg_latency,
                "peak_vram_mb": peak_vram
            }
        }
        
    def print_report(self, metrics: Dict[str, Any], title: str = "MedTwin AI Evaluation Report"):
        print("="*65)
        print(f" {title.upper()} ")
        print("="*65)
        print(f"Inference Latency: {metrics['system_telemetry']['avg_latency_ms']:.2f} ms/patient")
        print(f"Peak VRAM Memory:  {metrics['system_telemetry']['peak_vram_mb']:.2f} MB")
        print("-"*65)
        print(" CLINICAL METRIC     | BASE MODEL SYSTEM | HEALED MODEL SYSTEM | DELTA ")
        print("-"*65)
        for key in ["accuracy", "precision", "recall", "f1", "auroc", "ece"]:
            base_val = metrics["raw"][key]
            heal_val = metrics["healed"][key]
            delta = heal_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f" {key.capitalize():18} | {base_val:17.4f} | {heal_val:19.4f} | {sign}{delta:.4f}")
        print("-"*65)
        print("SELF-HEALING LOG SUMMARY:")
        for action, count in metrics["self_healing_telemetry"]["actions_triggered"].items():
            print(f" - {action:30} : {count} times")
        print(f"Conformal Empirical Coverage: {metrics['self_healing_telemetry']['conformal_empirical_coverage']:.2%}")
        print(f"Conformal Ambiguity Rate:     {metrics['self_healing_telemetry']['conformal_ambiguity_rate']:.2%}")
        print("="*65)
