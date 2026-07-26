import pytest
import numpy as np
import torch
from src.models.uncertainty import UncertaintyCalibrator

def test_ece_computation():
    # Perfect calibration mock: predictions match accuracies
    probs = np.array([0.1, 0.1, 0.2, 0.2, 0.8, 0.8, 0.9, 0.9])
    # 0.1s are 0 (accurate), 0.2s are 0 (accurate), 0.8s are 1 (accurate), 0.9s are 1 (accurate)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    
    ece = UncertaintyCalibrator.compute_ece(probs, labels, n_bins=10)
    assert ece >= 0.0
    # Perfect calibration should have low error
    assert ece < 0.20

def test_temperature_scaling():
    calibrator = UncertaintyCalibrator()
    logits = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32)
    labels = torch.tensor([0, 0, 1], dtype=torch.float32)
    
    # Fit temperature
    calibrator.fit_temperature(logits, labels, lr=0.1, epochs=10)
    
    # Check calibrated outputs
    probs = calibrator.get_calibrated_probability(logits)
    assert probs.shape == (3,)
    assert 0.0 <= probs[0].item() <= 1.0
    assert 0.0 <= probs[2].item() <= 1.0

def test_conformal_prediction():
    calibrator = UncertaintyCalibrator()
    
    # We include a misclassification in the validation set to increase threshold q,
    # ensuring both classes are included in the prediction set for ambiguous values like 0.48.
    val_probs = np.array([0.1, 0.2, 0.8, 0.55, 0.8, 0.9])
    val_labels = np.array([0, 0, 0, 1, 1, 1])
    
    # Fit conformal at 90% confidence (alpha=0.1)
    calibrator.fit_conformal(val_probs, val_labels, alpha=0.1)
    
    test_probs = np.array([0.05, 0.48, 0.95])
    sets = calibrator.predict_conformal_set(test_probs)
    
    assert len(sets) == 3
    # Highly confident clean should only contain class 0
    assert sets[0] == [0]
    # Highly confident tampered should only contain class 1
    assert sets[2] == [1]
    # Ambiguous/Uncertain score should contain both [0, 1] indicating ambiguity
    assert 0 in sets[1] and 1 in sets[1]
