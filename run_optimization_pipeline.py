import torch.nn as nn
from src.optimization.distributed import DistributedTrainingOrchestrator
from src.optimization.inference import InferenceOptimizer
from src.optimization.benchmarker import OptimizationBenchmarker

def run_optimization_pipeline():
    print("="*75)
    print(" MEDTWIN AI: DISTRIBUTED TRAINING & INFERENCE OPTIMIZATION PIPELINE ")
    print("="*75)
    
    # 1. Initialize Classes
    print("[1/4] Initializing orchestrator, optimizer, and benchmarker modules...")
    orchestrator = DistributedTrainingOrchestrator()
    optimizer = InferenceOptimizer()
    benchmarker = OptimizationBenchmarker()
    
    # 2. Print Distributed Params
    print("[2/4] Pulling distributed training parameter dictionaries...")
    ds_cfg = orchestrator.get_deepspeed_config(zero_stage=3)
    acc_cfg = orchestrator.get_accelerate_config()
    fsdp_rules = orchestrator.get_fsdp_wrapping_rules()
    
    print("\n" + "-"*65)
    print(" DISTRIBUTED CONFIGURATION SUMMARY ")
    print("-"*65)
    print(f"  * DeepSpeed Stage 3 Offload: {ds_cfg['zero_optimization']['offload_param']['device']}")
    print(f"  * Accelerate Process Count:   {acc_cfg['num_processes']}")
    print(f"  * FSDP Wrapping Layers:       {fsdp_rules['fsdp_transformer_layer_cls_to_wrap']}")
    print("-"*65)
    
    # 3. Simulate Model Optimizations
    print("\n[3/4] Running model compile and layer quantization wraps...")
    mock_model = nn.Sequential(nn.Linear(10, 10))
    
    orchestrator.enable_mixed_precision(mock_model, "bf16")
    orchestrator.enable_gradient_checkpointing(mock_model)
    
    optimizer.enable_flash_attention(mock_model)
    optimizer.quantize_int8(mock_model)
    optimizer.compile_model(mock_model)
    
    # 4. Inferences Benchmarks
    print("\n[4/4] Profiling execution speeds, throughput, and costs...")
    results = benchmarker.run_inference_benchmarks()
    
    print("\n" + "="*80)
    print(" RUNTIME INFERENCE OPTIMIZATION BENCHMARK SUMMARY ")
    print("="*80)
    print(" RUNTIME BACKEND  | LATENCY (ms) | VRAM MEM (GB) | THROUGHPUT | COST FACTOR ")
    print("-"*80)
    for k, v in results.items():
        print(f" {k:<16} | {v['latency_ms']:>12.1f} | {v['vram_gb']:>13.1f} | {v['throughput']:>10.1f} | {v['cost_ratio']:>11.2f}")
    print("="*80)
    
    benchmarker.plot_benchmarks(results, "docs/optimization_benchmarks.png")
    
    print("\nDistributed Training and Optimization pipeline complete.")

if __name__ == "__main__":
    run_optimization_pipeline()
