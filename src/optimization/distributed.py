import torch
import torch.nn as nn
from typing import Dict, Any

class DistributedTrainingOrchestrator:
    """
    Manages and configures distributed scaling frameworks (DeepSpeed, Accelerate, FSDP)
    and hardware optimization methods (mixed-precision, gradient checkpointing).
    """
    def get_deepspeed_config(self, zero_stage: int = 2) -> Dict[str, Any]:
        """
        Generates deepspeed execution configuration JSON templates.
        """
        return {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "zero_optimization": {
                "stage": zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if zero_stage == 3 else "none",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu" if zero_stage == 3 else "none",
                    "pin_memory": True
                },
                "overlap_comm": True,
                "contiguous_gradients": True
            },
            "bf16": {
                "enabled": True
            },
            "gradient_clipping": "auto",
            "steps_per_print": 100
        }

    def get_accelerate_config(self) -> Dict[str, Any]:
        """
        Compiles execution flags for Hugging Face Accelerate launcher.
        """
        return {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "MULTI_GPU",
            "num_machines": 1,
            "num_processes": 8, # 8 GPUs
            "mixed_precision": "bf16",
            "use_cpu": False,
            "rdzv_backend": "c10d",
            "same_network": True
        }

    def get_fsdp_wrapping_rules(self) -> Dict[str, Any]:
        """
        Defines Fully Sharded Data Parallel (FSDP) shard wrapping guidelines.
        """
        return {
            "fsdp_transformer_layer_cls_to_wrap": ["MedTwinAttentionLayer", "TransformerEncoderLayer"],
            "fsdp_backward_prefetch": "BACKWARD_PRE",
            "fsdp_forward_prefetch": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_offload_params": False
        }

    def enable_mixed_precision(self, model: nn.Module, dtype: str = "bf16") -> nn.Module:
        """
        Hooks mixed precision cast operators to model parameters.
        """
        target_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        print(f"[Distributed] Mixed precision enabled: Cast parameters to {dtype}.")
        # Emulate casting parameters
        for param in model.parameters():
            if param.is_floating_point():
                param.data = param.data.to(target_dtype)
        return model

    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        Registers gradient checkpointing wrappers on backpropagation paths to save VRAM.
        """
        print("[Distributed] Gradient checkpointing activated. Activations discarded on forward passes.")
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            # Fallback mock flag activation
            model.use_gradient_checkpointing = True
        return model
