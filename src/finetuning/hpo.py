from typing import Dict, Any, List
import torch.nn as nn
from torch.utils.data import DataLoader

from src.finetuning.trainer import ClinicalPEFTTrainer

class ClinicalHPOSearch:
    """
    Grid Search Optimizer for tuning SFT adapters (LoRA/QLoRA).
    Sweeps learning rates and ranks to identify convergence maximums.
    """
    def __init__(self, learning_rates: List[float] = [1e-4, 2e-4, 5e-4], lora_ranks: List[int] = [4, 8, 16]):
        self.learning_rates = learning_rates
        self.lora_ranks = lora_ranks

    def run_grid_search(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        checkpoint_dir: str = "checkpoints/hpo"
    ) -> Dict[str, Any]:
        """
        Executes grid search and returns best parameter metadata.
        """
        best_acc = -1.0
        best_config = {}
        trials_logged = []
        
        print(f"Starting HPO grid search sweeping: LRs={self.learning_rates}, Ranks={self.lora_ranks}...")
        
        import copy
        for lr in self.learning_rates:
            for rank in self.lora_ranks:
                model_trial = copy.deepcopy(model)
                # Instantiate trial trainer
                trainer = ClinicalPEFTTrainer(
                    model=model_trial,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    checkpoint_dir=checkpoint_dir,
                    learning_rate=lr,
                    lora_r=rank
                )
                
                # Apply adapter and run 1 trial epoch
                peft_model = trainer.apply_peft_adapter(config_type="lora")
                trial_res = trainer.train_epoch(config_type="lora")
                
                acc = trial_res["validation_accuracy"]
                f1 = trial_res["validation_f1"]
                
                trials_logged.append({
                    "lr": lr,
                    "lora_r": rank,
                    "accuracy": acc,
                    "f1": f1
                })
                
                print(f" - Trial: LR={lr:.0e} | Rank={rank:2d} | Val Acc={acc:.2%} | F1={f1:.4f}")
                
                if acc > best_acc:
                    best_acc = acc
                    best_config = {
                        "learning_rate": lr,
                        "lora_r": rank,
                        "accuracy": acc,
                        "f1": f1
                    }
                    
        print(f"Optimal configuration identified: LR={best_config['learning_rate']:.0e}, Rank={best_config['lora_r']}")
        return {
            "best_config": best_config,
            "trials": trials_logged
        }
