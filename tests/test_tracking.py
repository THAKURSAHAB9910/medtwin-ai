import pytest
import os

from src.tracking.tracker import ClinicalExperimentTracker
from src.tracking.dashboard import ExperimentDashboardGenerator

def test_experiment_tracker():
    tracker = ClinicalExperimentTracker()
    
    # 1. Start run
    run_id = tracker.start_run("test_rank_16")
    assert tracker.active_run_id == run_id
    assert tracker.mlflow_active is True
    
    # 2. Log parameters & metrics
    tracker.log_parameters({"lora_rank": 16, "learning_rate": 3e-4})
    tracker.log_metrics(step=1, metrics={"validation_f1": 0.92})
    
    # 3. Finalize run
    tracker.save_run("test_rank_16", {"lora_rank": 16}, {"validation_f1": 0.92})
    assert tracker.active_run_id is None
    assert len(tracker.runs_registry) == 1
    assert tracker.runs_registry[0]["run_name"] == "test_rank_16"

def test_dashboard_generator(tmp_path):
    generator = ExperimentDashboardGenerator()
    runs = [
        {"run_name": "run1", "parameters": {"lora_rank": 4}, "metrics": {"peak_vram_gb": 6.8, "validation_f1": 0.865}},
        {"run_name": "run2", "parameters": {"lora_rank": 8}, "metrics": {"peak_vram_gb": 12.5, "validation_f1": 0.910}}
    ]
    
    output_png = tmp_path / "test_dashboard.png"
    generator.generate_tracking_dashboard(runs, str(output_png))
    
    assert os.path.exists(output_png)
