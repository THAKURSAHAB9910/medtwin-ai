import os
import argparse
import pytorch_lightning as pl
import torch

from src.data.dataset import MedicalDataModule
from src.training.trainer import MedTwinAIModule
from src.training.self_healing import ClinicalSelfHealingEngine
from src.evaluation.evaluator import MedTwinAIEvaluator
from src.explanation.explain import ClinicalExplainer

def run_medtwin_pipeline(epochs: int = 2, batch_size: int = 4):
    print("="*75)
    print(" MEDTWIN AI: MULTIMODAL CLINICAL FOUNDATION SYSTEM LIFECYCLE ")
    print("="*75)
    
    # 1. Dataset setup
    print("[1/5] Synthesizing tri-modal patient records (PatientCraft)...")
    dm = MedicalDataModule(
        batch_size=batch_size,
        train_samples=40,
        val_samples=15,
        test_samples=15,
        image_size=(384, 384)
    )
    dm.setup()
    print(f"Generated train size: {len(dm.train_dataset)} patients")
    print(f"Generated validation size: {len(dm.val_dataset)} patients")
    print(f"Generated Out-Of-Distribution (OOD) test size: {len(dm.test_dataset)} patients")
    
    # 2. Model setup
    print("\n[2/5] Initializing Tri-Modal visual-textual-structured network...")
    model = MedTwinAIModule(
        backbone_name="resnet18",
        embed_dim=256,
        use_mock_vlm=True,
        learning_rate=1e-3
    )
    
    # 3. Model Training
    print("\n[3/5] Starting Multi-Task Training Loop...")
    # Enable lightweight CPU/GPU execution without logging overhead
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True
    )
    trainer.fit(model, datamodule=dm)
    print("Training loop finished successfully.")
    
    # 4. Evaluation and Calibration Benchmarking
    print("\n[4/5] Running Clinical Calibration and Benchmarking...")
    self_healing_engine = ClinicalSelfHealingEngine(uncertainty_threshold=0.35, lesion_threshold=0.1)
    evaluator = MedTwinAIEvaluator(model, self_healing_engine)
    
    # In-Distribution Validation Evaluation
    val_loader = dm.val_dataloader()
    val_report = evaluator.evaluate_dataset(val_loader)
    evaluator.print_report(val_report, title="Validation Set (In-Distribution) Report")
    
    # Out-Of-Distribution Test Evaluation (High noise scans)
    test_loader = dm.test_dataloader()
    test_report = evaluator.evaluate_dataset(test_loader)
    evaluator.print_report(test_report, title="OOD Test Set (High Noise scans) Report")
    
    # 5. Diagnostic Explanations
    print("\n[5/5] Generating Visual Diagnostic Explanations...")
    explainer = ClinicalExplainer(output_dir="docs")
    
    # Grab a diseased patient case from validation set
    val_dataset = dm.val_dataset
    diseased_sample = None
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        if sample["label"].item() == 1:
            diseased_sample = sample
            break
            
    if diseased_sample is not None:
        img_tensor = diseased_sample["image"].unsqueeze(0).to(model.device)
        notes = [diseased_sample["clinical_note"]]
        labs_tensor = diseased_sample["lab_metrics"].unsqueeze(0).to(model.device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(img_tensor, notes, labs_tensor)
            
        explanation_path = explainer.explain_sample(
            image_tensor=diseased_sample["image"],
            lesion_tensor=outputs["lesion_map"].squeeze(0),
            metadata=diseased_sample["metadata"],
            clinical_note=diseased_sample["clinical_note"],
            save_name="medtwin_clinical_explanation_sample.png"
        )
        print(f"Verification complete. Diagnostic multi-panel saved: {explanation_path}")
    else:
        print("Warning: No pneumonia sample found in validation set to explain.")
        
    print("\n" + "="*75)
    print(" RESEARCH PIPELINE COMPLETED SUCCESSFULLY ")
    print("="*75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedTwin AI Research Pipeline")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    args = parser.parse_args()
    
    run_medtwin_pipeline(epochs=args.epochs, batch_size=args.batch_size)
