import pytest
import torch
import numpy as np

from src.twin.twin_model import BayesianPatientTwinModel
from src.twin.simulation import PatientDigitalTwinEngine
from src.evaluation.twin_evaluator import PatientTwinEvaluator

def test_bayesian_patient_twin_model():
    model = BayesianPatientTwinModel(state_dim=6, action_dim=5)
    
    # 1. Test single step transition
    state = torch.randn(2, 6)
    action = torch.randn(2, 5)
    next_state = model(state, action)
    assert next_state.shape == (2, 6)
    
    # 2. Test Monte Carlo rollout
    s0 = torch.tensor([[80.0, 130.0, 120.0, 75.0, 7500.0, 97.0]], dtype=torch.float32)
    actions = torch.tensor([[-0.5, 500.0, 0.0, 3.0, 2.0]] * 12, dtype=torch.float32)
    
    means, stds = model.rollout_simulation(s0, actions, steps=12, num_mc_samples=10)
    assert means.shape == (13, 6)
    assert stds.shape == (13, 6)
    assert np.all(stds >= 0.0)

def test_patient_digital_twin_engine():
    engine = PatientDigitalTwinEngine()
    
    vitals = {
        "weight": 90.0,
        "blood_pressure": 140.0,
        "blood_glucose": 160.0,
        "heart_rate": 80.0,
        "wbc_count": 8000.0,
        "oxygen_saturation": 96.0
    }
    
    scenario = {
        "weight_change": -8.0,
        "metformin_mg": 500.0,
        "beta_blocker_mg": 25.0,
        "exercise_hours": 3.0,
        "sodium_intake_g": 1.5
    }
    
    res = engine.run_what_if_simulation(vitals, scenario, steps=6)
    
    assert "trajectories" in res
    assert len(res["trajectories"]["weight"]["mean"]) == 7
    assert 0.0 <= res["risk_scores"]["cardiovascular_risk"] <= 1.0
    assert 0.0 <= res["risk_scores"]["hospital_readmission_risk"] <= 1.0
    assert res["risk_scores"]["predicted_recovery_time_days"] > 0
    assert len(res["guideline_citations"]) > 0

def test_patient_twin_evaluator():
    evaluator = PatientTwinEvaluator()
    
    predicted = {
        "weight": {"mean": [80.0, 79.0, 78.0], "std": [0.5, 0.5, 0.5]},
        "blood_pressure": {"mean": [120.0, 118.0, 116.0], "std": [1.0, 1.0, 1.0]},
        "blood_glucose": {"mean": [140.0, 138.0, 136.0], "std": [2.0, 2.0, 2.0]}
    }
    
    actual = {
        "weight": [80.0, 79.1, 78.2],
        "blood_pressure": [120.0, 117.8, 116.3],
        "blood_glucose": [140.0, 138.5, 136.1]
    }
    
    metrics = evaluator.evaluate_trajectories(predicted, actual)
    
    assert "weight" in metrics
    assert "mae" in metrics["weight"]
    assert 0.0 <= metrics["weight"]["coverage_95"] <= 1.0
