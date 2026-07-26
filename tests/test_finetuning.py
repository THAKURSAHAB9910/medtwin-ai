import pytest
import os
import torch
import torch.nn as nn

from src.data.generator import PatientCraftEngine
from src.finetuning.dataset import ClinicalSFTDataset, MockTokenizer, create_sft_dataloader
from src.finetuning.trainer import ClinicalPEFTTrainer
from src.finetuning.hpo import ClinicalHPOSearch

class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5, 2)
    def forward(self, x):
        return self.lin(x)

def test_sft_dataset_formatting():
    engine = PatientCraftEngine(seed=45)
    records = [engine.generate_patient_record(has_pneumonia=True)]
    
    tokenizer = MockTokenizer()
    dataset = ClinicalSFTDataset(records, tokenizer, max_length=128)
    
    assert len(dataset) == 1
    item = dataset[0]
    
    assert "input_ids" in item
    assert "labels" in item
    assert "pixel_values" in item
    assert item["input_ids"].shape[0] == 128
    assert item["labels"].shape[0] == 128

def test_peft_adapter_injection():
    model = SimpleTestModel()
    engine = PatientCraftEngine(seed=45)
    records = [engine.generate_patient_record(has_pneumonia=True) for _ in range(4)]
    tokenizer = MockTokenizer()
    
    loader = create_sft_dataloader(records, tokenizer, batch_size=2)
    
    trainer = ClinicalPEFTTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        checkpoint_dir="checkpoints/test_peft"
    )
    
    peft_model = trainer.apply_peft_adapter(config_type="lora")
    assert peft_model is not None

def test_trainer_epoch_and_checkpoint():
    model = SimpleTestModel()
    engine = PatientCraftEngine(seed=45)
    records = [engine.generate_patient_record(has_pneumonia=True) for _ in range(4)]
    tokenizer = MockTokenizer()
    loader = create_sft_dataloader(records, tokenizer, batch_size=2)
    
    chk_dir = "checkpoints/test_chk"
    trainer = ClinicalPEFTTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        checkpoint_dir=chk_dir
    )
    
    # Run epoch training
    res = trainer.train_epoch(config_type="lora")
    assert res["epoch"] == 1
    assert "avg_loss" in res
    assert res["validation_accuracy"] > 0.0
    
    # Verify checkpoint output
    chk_path = os.path.join(chk_dir, "checkpoint-step-2")
    assert os.path.exists(chk_path)
    assert os.path.exists(os.path.join(chk_path, "trainer_state.json"))
    assert os.path.exists(os.path.join(chk_path, "adapter_model.bin"))
    
    # Test resume training
    new_trainer = ClinicalPEFTTrainer(
        model=SimpleTestModel(),
        train_loader=loader,
        val_loader=loader,
        checkpoint_dir=chk_dir
    )
    success = new_trainer.resume_training(chk_path)
    assert success
    assert new_trainer.current_epoch == 1
    assert new_trainer.global_step == 2
    
    # Cleanup files
    if os.path.exists(os.path.join(chk_path, "trainer_state.json")):
        os.remove(os.path.join(chk_path, "trainer_state.json"))
    if os.path.exists(os.path.join(chk_path, "adapter_model.bin")):
        os.remove(os.path.join(chk_path, "adapter_model.bin"))
    if os.path.exists(chk_path):
        os.rmdir(chk_path)

def test_hpo_search_sweep():
    model = SimpleTestModel()
    engine = PatientCraftEngine(seed=45)
    records = [engine.generate_patient_record(has_pneumonia=True) for _ in range(4)]
    tokenizer = MockTokenizer()
    loader = create_sft_dataloader(records, tokenizer, batch_size=2)
    
    # Run sweep with smaller grid to keep tests quick
    hpo = ClinicalHPOSearch(learning_rates=[1e-4], lora_ranks=[8])
    res = hpo.run_grid_search(model, loader, loader, checkpoint_dir="checkpoints/test_hpo")
    
    assert "best_config" in res
    assert "trials" in res
    assert len(res["trials"]) == 1
    assert res["best_config"]["lora_r"] == 8
