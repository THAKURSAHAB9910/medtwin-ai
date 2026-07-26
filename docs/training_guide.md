# MedTwin AI: Training Guide

## 1. Environment Configurations
Prepare PyTorch 2.0+ and CUDA execution toolkits. Install required dependencies:
```bash
pip install torch transformers accelerate deepspeed peft
```

---

## 2. LoRA Fine-Tuning Execution
To fine-tune MedTwin AI modular adapters (PEFT) on proprietary custom hospital dataset folders:

```bash
python src/finetuning/trainer.py --data_dir ./data/train --epochs 5 --lr 3e-4 --lora_rank 16
```

### Core Parameters
* `--lora_rank`: Defines attention mapping low-rank dimension parameter ($r = 16$).
* `--lr`: Optimizer learning rate ($3 \times 10^{-4}$).
* `--epochs`: Training iteration steps.

---

## 3. DeepSpeed & Mixed-Precision Scaling
To scale training across multiple GPUs, run under DeepSpeed ZeRO-3 parameter offloading:

```bash
deepspeed --num_gpus=8 src/finetuning/trainer.py \
    --deepspeed_config ./configs/ds_stage3.json \
    --bf16 True
```
* **ZeRO Stage 3**: Offloads optimizer states, parameters, and gradients to host CPU memory when GPU VRAM limits are exceeded.
* **Mixed-Precision**: Casts floating operations to `bfloat16` to reduce tensor memory allocation by 50%.
