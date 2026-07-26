import torch
import numpy as np
from typing import Dict, Any, List

class ClinicalSelfHealingEngine:
    """
    Clinical Self-Healing and Active Reasoning Engine.
    Performs cross-modal physiological logic auditing when predictions are ambiguous.
    Evaluates vitals constraints (temperature, leukocytosis, blood oxygenation) to override
    model hallucinations and calibrate critical diagnostics.
    """
    def __init__(
        self,
        uncertainty_threshold: float = 0.35,
        fever_limit: float = 100.4,
        wbc_limit: float = 11000.0,
        hypoxemia_limit: float = 95.0,
        lesion_threshold: float = 0.25
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.fever_limit = fever_limit
        self.wbc_limit = wbc_limit
        self.hypoxemia_limit = hypoxemia_limit
        self.lesion_threshold = lesion_threshold

    def run_physiological_audit(self, lab_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Applies clinical diagnostic rules over numerical patient metrics.
        Returns:
            has_infection_evidence: True if physiological vitals strongly indicate infection
            has_healthy_evidence: True if physiological vitals strongly indicate a healthy profile
        """
        temp = lab_metrics.get("temperature", 98.6)
        wbc = lab_metrics.get("wbc_count", 7500.0)
        spo2 = lab_metrics.get("oxygen_saturation", 98.0)
        
        # Clinical criteria for systemic infection (Pneumonia symptoms)
        fever = temp >= self.fever_limit
        leukocytosis = wbc >= self.wbc_limit
        hypoxemia = spo2 <= self.hypoxemia_limit
        
        # Strong infection evidence: at least two matching clinical markers
        has_infection_evidence = (fever and leukocytosis) or (fever and hypoxemia) or (leukocytosis and hypoxemia)
        
        # Healthy evidence: normal temp, normal wbc, normal oxygen saturation
        has_healthy_evidence = (temp < 99.2) and (wbc < 10000.0) and (spo2 >= 96.0)
        
        return {
            "fever": fever,
            "leukocytosis": leukocytosis,
            "hypoxemia": hypoxemia,
            "has_infection_evidence": has_infection_evidence,
            "has_healthy_evidence": has_healthy_evidence,
            "details": f"Temp: {temp}F, WBC: {wbc}/uL, SpO2: {spo2}%"
        }

    def run_lesion_verification(self, lesion_map: np.ndarray) -> Dict[str, Any]:
        """
        Analyzes the predicted lesion segmentation map to calculate visual pathology density.
        """
        # Mean intensity of predicted pathology map
        mean_pathology_score = float(np.mean(lesion_map))
        has_visual_lesion = mean_pathology_score >= self.lesion_threshold
        
        return {
            "mean_pathology_score": mean_pathology_score,
            "has_visual_lesion": has_visual_lesion
        }

    def verify_and_heal(
        self,
        calibrated_prob: float,
        lesion_map: np.ndarray,
        lab_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Executes clinical self-healing logical checks.
        Args:
            calibrated_prob: Probability outputted from calibrated fusion network
            lesion_map: Segmented lesion prediction mask [H, W]
            lab_metrics: Numerical vital parameters
        """
        margin = abs(calibrated_prob - 0.5)
        is_uncertain = margin < self.uncertainty_threshold
        
        phys_audit = self.run_physiological_audit(lab_metrics)
        visual_audit = self.run_lesion_verification(lesion_map)
        
        healed_prob = calibrated_prob
        action_taken = "none"
        reason = "Diagnostic prediction was confident; bypassed healing."
        
        if is_uncertain:
            reason = "Diagnostic prediction was clinically ambiguous. Triggering self-healing..."
            
            # Healing Case 1: High uncertainty, but vitals show clear healthy profile
            if phys_audit["has_healthy_evidence"] and not visual_audit["has_visual_lesion"]:
                healed_prob = 0.0 # Clear healthy override
                action_taken = "healed_normal_verification"
                reason += f" Physiological check indicates clear healthy profile: {phys_audit['details']} with low lesion density ({visual_audit['mean_pathology_score']:.4f}). Overridden to Normal."
                
            # Healing Case 2: High uncertainty, but vitals show infection and visual lesions are present
            elif phys_audit["has_infection_evidence"] or visual_audit["has_visual_lesion"]:
                healed_prob = 1.0 # Clear disease override
                action_taken = "healed_infection_verification"
                reason += f" High physiological or visual disease markers detected: {phys_audit['details']}, pathology density: {visual_audit['mean_pathology_score']:.4f}. Overridden to Pneumonia."
                
            # Healing Case 3: Conflicting or intermediate -> default to cautious routing
            else:
                healed_prob = calibrated_prob
                action_taken = "routed_to_physician"
                reason += " Cross-modal verification remains ambiguous due to conflicting lab-visual markers. Routed to physician review."
                
        healed_label = int(healed_prob >= 0.5)
        
        return {
            "original_prob": calibrated_prob,
            "healed_prob": healed_prob,
            "healed_label": healed_label,
            "action_taken": action_taken,
            "reason": reason,
            "physiological_check": phys_audit,
            "visual_check": visual_audit
        }

if __name__ == "__main__":
    engine = ClinicalSelfHealingEngine()
    
    # Test healthy profile correction
    vitals_healthy = {"temperature": 98.2, "wbc_count": 5500.0, "oxygen_saturation": 98.0}
    lesion_empty = np.zeros((128, 128))
    res1 = engine.verify_and_heal(0.53, lesion_empty, vitals_healthy)
    print("Self Healing (Healthy Check):")
    print(f" - Action: {res1['action_taken']}")
    print(f" - Prob: {res1['original_prob']} -> {res1['healed_prob']}")
    print(f" - Reason: {res1['reason']}")
    
    # Test infection profile correction
    vitals_sick = {"temperature": 102.5, "wbc_count": 15000.0, "oxygen_saturation": 89.0}
    res2 = engine.verify_and_heal(0.47, lesion_empty, vitals_sick)
    print("\nSelf Healing (Infection Check):")
    print(f" - Action: {res2['action_taken']}")
    print(f" - Prob: {res2['original_prob']} -> {res2['healed_prob']}")
    print(f" - Reason: {res2['reason']}")
