import pytest
import os
import torch.nn as nn

from src.optimization.distributed import DistributedTrainingOrchestrator
from src.optimization.inference import InferenceOptimizer
from src.optimization.benchmarker import OptimizationBenchmarker

def test_distributed_orchestrator():
    orchestrator = DistributedTrainingOrchestrator()
    
    # 1. DeepSpeed Config
    ds_cfg = orchestrator.get_deepspeed_config(zero_stage=3)
    assert ds_cfg["zero_optimization"]["stage"] == 3
    assert ds_cfg["zero_optimization"]["offload_param"]["device"] == "cpu"
    
    # 2. Accelerate Config
    acc_cfg = orchestrator.get_accelerate_config()
    assert acc_cfg["mixed_precision"] == "bf16"
    assert acc_cfg["num_processes"] == 8
    
    # 3. FSDP Rules
    fsdp = orchestrator.get_fsdp_wrapping_rules()
    assert "TransformerEncoderLayer" in fsdp["fsdp_transformer_layer_cls_to_wrap"]
    
    # 4. Mixed precision casting
    model = nn.Sequential(nn.Linear(10, 10))
    orchestrator.enable_mixed_precision(model, "bf16")
    orchestrator.enable_gradient_checkpointing(model)
    assert hasattr(model, "use_gradient_checkpointing") or hasattr(model, "gradient_checkpointing_enable")

def test_inference_optimizer():
    optimizer = InferenceOptimizer()
    model = nn.Sequential(nn.Linear(10, 10))
    
    # ONNX conversion metadata
    onnx_info = optimizer.convert_to_onnx(model, "dummy.onnx")
    assert onnx_info["opset_version"] == 17
    
    # TensorRT conversion metadata
    trt_info = optimizer.convert_to_tensorrt(onnx_info, "dummy.engine")
    assert trt_info["precision"] == "FP16"
    
    # INT8 Dynamic Quantization
    quant_model = optimizer.quantize_int8(model)
    assert hasattr(quant_model, "is_quantized_int8") or "Quantized" in str(quant_model)
    
    # Flash attention hook
    optimizer.enable_flash_attention(model)
    assert model.use_flash_attention is True

def test_optimization_benchmarker(tmp_path):
    benchmarker = OptimizationBenchmarker()
    results = benchmarker.run_inference_benchmarks()
    
    assert "TensorRT" in results
    assert "FP32 Baseline" in results
    assert results["TensorRT"]["latency_ms"] == 12.0
    
    out_png = tmp_path / "test_bench.png"
    benchmarker.plot_benchmarks(results, str(out_png))
    assert os.path.exists(out_png)
