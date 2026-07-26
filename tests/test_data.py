import pytest
import numpy as np
import torch
from src.data.generator import PatientCraftEngine
from src.data.dataset import MedicalMultimodalDataset

def test_patient_craft_engine():
    engine = PatientCraftEngine(seed=42)
    
    # 1. Test healthy patient record
    sample_normal = engine.generate_patient_record(has_pneumonia=False)
    assert sample_normal["image"].size == (512, 512)
    assert sample_normal["mask"].size == (512, 512)
    assert np.max(np.array(sample_normal["mask"])) == 0 # Normal has no lesion
    assert sample_normal["label"] == 0
    assert sample_normal["lab_metrics"]["temperature"] < 99.5
    assert sample_normal["lab_metrics"]["oxygen_saturation"] >= 95
    
    # 2. Test infection patient record
    sample_sick = engine.generate_patient_record(has_pneumonia=True)
    assert sample_sick["image"].size == (512, 512)
    assert sample_sick["mask"].size == (512, 512)
    assert np.max(np.array(sample_sick["mask"])) == 255 # Pneumonia has lesion
    assert sample_sick["label"] == 1
    assert sample_sick["lab_metrics"]["temperature"] >= 100.0
    assert sample_sick["lab_metrics"]["wbc_count"] >= 12000

def test_medical_dataset():
    dataset = MedicalMultimodalDataset(
        num_samples=4,
        is_training=True,
        pneumonia_prob=0.5,
        image_size=(256, 256),
        seed=12
    )
    
    assert len(dataset) == 4
    sample = dataset[0]
    
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["lab_metrics"].shape == (5,) # [age, temp, hr, wbc, spo2]
    assert sample["label"].item() in [0, 1]
    assert isinstance(sample["clinical_note"], str)
