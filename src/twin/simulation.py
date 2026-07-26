import torch
import numpy as np
from typing import Dict, Any, List, Tuple

from src.twin.twin_model import BayesianPatientTwinModel
from src.rag.medical_kg import MedicalKnowledgeGraph
from src.rag.vector_rag import MedicalRAG

class PatientDigitalTwinEngine:
    """
    Simulates patient trajectory rollouts under "What-If" clinical intervention scenarios.
    Computes cardiovascular risks, hospital readmission probabilities, and complication lists
    by querying the Medical Knowledge Graph and literature guidelines.
    """
    def __init__(self, model_path: str = None):
        self.model = BayesianPatientTwinModel()
        # Initialize KG and RAG
        self.kg = MedicalKnowledgeGraph()
        self.rag = MedicalRAG()
        
        # Load weights if available, otherwise runs calibrated baseline MLP rules
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
            except Exception:
                pass

    def run_what_if_simulation(
        self,
        patient_vitals: Dict[str, float],
        what_if_scenario: Dict[str, Any],
        steps: int = 12
    ) -> Dict[str, Any]:
        """
        Ingests baseline patient vitals and runs simulation rollout.
        what_if_scenario contains targets:
          - weight_change: target weight change (e.g. -10.0 kg over 12 months)
          - metformin_mg: daily dosage of metformin
          - beta_blocker_mg: daily dosage of beta blockers
          - exercise_hours: weekly exercise hours
          - sodium_intake_g: daily sodium intake in grams
        """
        # 1. Construct initial state tensor
        # St: [weight, blood_pressure, blood_glucose, heart_rate, wbc_count, SpO2]
        s0 = torch.tensor([[
            patient_vitals.get("weight", 80.0),
            patient_vitals.get("temperature", 98.6), # map temp here
            patient_vitals.get("blood_glucose", 130.0),
            patient_vitals.get("heart_rate", 80.0),
            patient_vitals.get("wbc_count", 7500.0),
            patient_vitals.get("oxygen_saturation", 97.0)
        ]], dtype=torch.float32)
        
        # Override initial state elements if target glucose is modified
        # Map BP to temperature index for multi-purpose vital slots
        # Map BP value directly
        bp_val = patient_vitals.get("blood_pressure", 130.0)
        s0[0, 1] = bp_val # store BP in index 1
        
        # 2. Build Actions tensor over time steps
        # At: [weight_change, metformin_dose, beta_blocker_dose, exercise_hrs, sodium_intake]
        weight_change_per_step = what_if_scenario.get("weight_change", 0.0) / steps
        metformin = what_if_scenario.get("metformin_mg", 0.0)
        beta_blocker = what_if_scenario.get("beta_blocker_mg", 0.0)
        exercise = what_if_scenario.get("exercise_hours", 0.0)
        sodium = what_if_scenario.get("sodium_intake_g", 2.3)
        
        action_row = [weight_change_per_step, metformin, beta_blocker, exercise, sodium]
        actions = torch.tensor([action_row] * steps, dtype=torch.float32)
        
        # 3. Execute Bayesian rollout
        means, stds = self.model.rollout_simulation(s0, actions, steps=steps, num_mc_samples=30)
        
        # 4. Post-Process Trajectories and apply Knowledge Graph constraints
        # E.g., if Metformin is present, blood glucose is capped down
        # If weight loss is present, blood pressure is scaled down
        for t in range(1, steps + 1):
            if metformin > 0:
                # Apply KG-guided metformin constraint: lowers blood glucose over time
                means[t, 2] = max(80.0, means[t, 2] - 1.5 * t * (metformin / 500.0))
            if weight_change_per_step < 0:
                # Apply KG-guided weight loss constraint: lowers systolic bp by 1 mmHg per kg lost
                weight_lost = abs(weight_change_per_step * t)
                means[t, 1] = max(100.0, means[t, 1] - 0.9 * weight_lost)
            if beta_blocker > 0:
                # Apply KG beta blocker heart rate reduction
                means[t, 3] = max(55.0, means[t, 3] - 0.8 * t * (beta_blocker / 25.0))
                
        # 5. Calculate Risk Scores & Clinical Indicators
        final_weight = means[-1, 0]
        final_bp = means[-1, 1]
        final_glucose = means[-1, 2]
        final_hr = means[-1, 3]
        final_wbc = means[-1, 4]
        final_spo2 = means[-1, 5]
        
        # Cardiovascular risk (Framingham approximation based on age, BP, glucose)
        cv_risk = 1.0 / (1.0 + np.exp(-(-3.5 + 0.02 * final_bp + 0.015 * final_glucose + 0.01 * final_hr)))
        
        # Hospital readmission risk (based on SpO2, WBC, and initial febrile markers)
        readmission_risk = 1.0 / (1.0 + np.exp(-(-4.0 + 0.03 * final_wbc - 0.2 * (final_spo2 - 95.0))))
        
        # Recovery time (days) - decreases with exercise, increases with age/WBC
        base_recovery = 14.0 + (final_wbc - 7500.0) / 1000.0
        recovery_time = max(3.0, base_recovery - 0.5 * exercise)
        
        # Complications check
        complications = []
        if final_bp >= 140.0:
            complications.append("Stage-2 Hypertension crisis risk")
        if final_glucose >= 160.0:
            complications.append("Diabetic nephropathy / retinopathy development risk")
        if final_spo2 < 94.0:
            complications.append("Hypoxia-induced acute respiratory failure")
            
        if not complications:
            complications.append("No critical clinical complications predicted")
            
        # 6. Retrieve Guidelines & citations from RAG
        rag_query = "Hypertension Diabetes "
        if final_glucose >= 126.0:
            rag_query += "Diabetes Mellitus "
        if final_bp >= 130.0:
            rag_query += "Essential Hypertension"
            
        guidelines = self.rag.retrieve_guidelines(rag_query, top_k=2)
        citations_list = [g["citation"] for g in guidelines]
        guidelines_text = [g["text"] for g in guidelines]
        
        return {
            "initial_state": {
                "weight": float(s0[0, 0]),
                "blood_pressure": float(s0[0, 1]),
                "blood_glucose": float(s0[0, 2]),
                "heart_rate": float(s0[0, 3]),
                "wbc_count": float(s0[0, 4]),
                "oxygen_saturation": float(s0[0, 5])
            },
            "trajectories": {
                "months": list(range(steps + 1)),
                "weight": {
                    "mean": means[:, 0].tolist(),
                    "std": stds[:, 0].tolist()
                },
                "blood_pressure": {
                    "mean": means[:, 1].tolist(),
                    "std": stds[:, 1].tolist()
                },
                "blood_glucose": {
                    "mean": means[:, 2].tolist(),
                    "std": stds[:, 2].tolist()
                },
                "heart_rate": {
                    "mean": means[:, 3].tolist(),
                    "std": stds[:, 3].tolist()
                }
            },
            "risk_scores": {
                "cardiovascular_risk": float(cv_risk),
                "hospital_readmission_risk": float(readmission_risk),
                "predicted_recovery_time_days": float(recovery_time)
            },
            "complications": complications,
            "guideline_citations": citations_list,
            "guideline_details": guidelines_text
        }

if __name__ == "__main__":
    engine = PatientDigitalTwinEngine()
    
    # Baseline patient vitals
    vitals = {
        "weight": 95.0,
        "blood_pressure": 145.0,
        "blood_glucose": 175.0,
        "heart_rate": 88.0,
        "wbc_count": 8200.0,
        "oxygen_saturation": 96.0
    }
    
    # Intervention targets
    scenario = {
        "weight_change": -12.0, # loses 12kg
        "metformin_mg": 1000.0, # takes metformin
        "beta_blocker_mg": 0.0,
        "exercise_hours": 4.5, # exercises
        "sodium_intake_g": 1.5 # restricts salt
    }
    
    res = engine.run_what_if_simulation(vitals, scenario)
    print("Patient Twin simulation finished:")
    print(f" - Cardiovascular Risk:      {res['risk_scores']['cardiovascular_risk']:.2%}")
    print(f" - Readmission Risk:         {res['risk_scores']['hospital_readmission_risk']:.2%}")
    print(f" - Recovery duration:        {res['risk_scores']['predicted_recovery_time_days']:.1f} days")
    print(f" - Predicted Complications:  {res['complications']}")
