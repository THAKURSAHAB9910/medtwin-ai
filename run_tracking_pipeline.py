from src.tracking.tracker import ClinicalExperimentTracker
from src.tracking.dashboard import ExperimentDashboardGenerator

def run_experiment_tracking():
    print("="*75)
    print(" MEDTWIN AI: EXPERIMENT TRACKING PIPELINE (MLflow / W&B) ")
    print("="*75)
    
    # 1. Initialize Tracker
    print("[1/3] Initializing experiment tracker...")
    tracker = ClinicalExperimentTracker()
    dashboard = ExperimentDashboardGenerator()
    
    # 2. Simulate 4 training config runs
    print("[2/3] Simulating 4 hyperparameter runs (varying LoRA ranks)...")
    
    run_configs = [
        {"run_name": "LoRA_Rank_4", "params": {"lora_rank": 4, "learning_rate": 1e-4, "model_version": "v1.0", "dataset_version": "v1.2", "epochs": 5}, "metrics": {"peak_vram_gb": 6.8, "validation_f1": 0.865, "latency_ms": 38.0, "hallucination_rate": 0.380}},
        {"run_name": "LoRA_Rank_8", "params": {"lora_rank": 8, "learning_rate": 2e-4, "model_version": "v1.1", "dataset_version": "v1.2", "epochs": 5}, "metrics": {"peak_vram_gb": 12.5, "validation_f1": 0.910, "latency_ms": 42.0, "hallucination_rate": 0.220}},
        {"run_name": "LoRA_Rank_16", "params": {"lora_rank": 16, "learning_rate": 3e-4, "model_version": "v1.2", "dataset_version": "v1.2", "epochs": 5}, "metrics": {"peak_vram_gb": 16.2, "validation_f1": 0.945, "latency_ms": 45.0, "hallucination_rate": 0.120}},
        {"run_name": "LoRA_Rank_32", "params": {"lora_rank": 32, "learning_rate": 3e-4, "model_version": "v1.3", "dataset_version": "v1.2", "epochs": 5}, "metrics": {"peak_vram_gb": 22.0, "validation_f1": 0.948, "latency_ms": 52.0, "hallucination_rate": 0.080}}
    ]
    
    for cfg in run_configs:
        print(f"\n --- EXECUTING RUN: {cfg['run_name'].upper()} ---")
        tracker.start_run(cfg["run_name"])
        tracker.log_parameters(cfg["params"])
        
        # Simulate step logs
        tracker.log_metrics(step=1, metrics={"training_loss": 0.88, "vram_gb": cfg["metrics"]["peak_vram_gb"] * 0.9})
        tracker.log_metrics(step=5, metrics={"training_loss": 0.12, "vram_gb": cfg["metrics"]["peak_vram_gb"]})
        
        # Finalize and save
        tracker.save_run(cfg["run_name"], cfg["params"], cfg["metrics"])
        
    # 3. Compile and plot visual dashboard
    print("\n[3/3] Generating visual Weights & Biases dashboard plot...")
    dashboard.generate_tracking_dashboard(tracker.runs_registry, "docs/experiment_tracking_dashboard.png")
    
    print("\nExperiment tracking pipeline complete.")

if __name__ == "__main__":
    run_experiment_tracking()
