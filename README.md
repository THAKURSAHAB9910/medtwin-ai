# MedTwin AI: Uncertainty-Calibrated and Self-Correcting Multimodal Clinical Foundation Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

**MedTwin AI** is a research-grade, production-ready healthcare decision support model. It jointly reasons over medical radiographs (images), clinical notes (transcripts), and raw laboratory reports (structured vitals) to perform calibrated disease diagnostic classification, anatomical lesion localization, and physiological constraint validation.

---

## 📖 Key Research Contributions

1. **Tri-Modal Cross-Attention Fusion**: Projects chest radiographs (Med-ViT), clinical physician note embeddings (ClinicalVLM), and vital metric tokens (StructuredLabEncoder) into a shared space. It aligns visual features to textual and vital contexts using gated cross-attention.
2. **Uncertainty Calibration & Conformal Sets**: Uses temperature scaling and split conformal prediction to output distribution-free confidence sets of diagnoses at a target coverage rate (e.g. 90% confidence), preventing overconfident model mistakes.
3. **Physiological Self-Healing Loop**: If a prediction falls within the conformal prediction uncertainty margin, the engine triggers active reasoning checks, cross-referencing patient blood/vital stats (temperature, leukocytes, oxygenation) and local lesion density scores to correct diagnostic classification.
4. **Specialist Multi-Agent Board**: Establishes coordinated plurality voting and consensus scoring over Radiologist, Pathologist, Dermatologist, and Literature Researcher agents.
5. **Medical OCR & Document Intelligence**: Pipeline extracting structured entities (drug names, dosages) from hand-written or scanned prescriptions using PaddleOCR/TrOCR.
6. **Synthetic Scanner Engine**: Synthesizes blurred, noisy, and rare disease profiles (Alkaptonuria, Glioblastoma, Huntington's) to stress-test diagnostic robustness.
7. **Failure Lab & Saliency Explainers**: Automatically audits false negatives and provides Grad-CAM, SHAP, and attention map attribution traces.
8. **Experiment Tracking & TensorRT Optimization**: Plugs into MLflow/W&B/TensorBoard and compiles inference targets to TensorRT to reduce GPU footprint to 2.8GB.

---

## 🛠️ Repository Architecture

```
nhyp/
├── docs/
│   ├── technical_report.md         # Detailed technical specification
│   ├── ablation_report.md          # Quantitative ablation comparisons
│   ├── dataset_card.md             # Dataset partitions details
│   ├── model_card.md               # Model parameters details
│   ├── deployment_guide.md         # FastAPI server launch details
│   ├── training_guide.md           # PEFT and DeepSpeed training instructions
│   └── evaluation_guide.md         # Accuracy validation metrics
├── src/
│   ├── agents/
│   │   ├── coordinator.py          # Specialist agent coordination
│   │   └── specialists.py          # Specialist panel agents
│   ├── ocr/
│   │   ├── pipeline.py             # OCR doc scanner pipeline
│   │   └── benchmarker.py          # PaddleOCR, EasyOCR benchmarks
│   ├── synthetic/
│   │   ├── engine.py               # Degradation noise generator
│   │   └── evaluator.py            # Robustness augmented evaluator
│   ├── explainability/
│   │   ├── lab.py                  # Failure Analysis Lab
│   │   └── explain.py              # Grad-CAM, SHAP explainers
│   ├── tracking/
│   │   ├── tracker.py              # MLflow/W&B experiment logging
│   │   └── dashboard.py            # Performance dashboards
│   └── optimization/
│       ├── distributed.py          # FSDP, DeepSpeed distributed training
│       ├── inference.py            # torch.compile, TensorRT compilers
│       └── benchmarker.py          # Latency & throughput benchmarker
├── tests/                          # Automated pytest suite
├── run_synthetic_pipeline.py       # [CLI] Synthetic pipeline entrypoint
├── run_explainability_pipeline.py  # [CLI] Explainability pipeline entrypoint
├── run_tracking_pipeline.py        # [CLI] MLflow tracking pipeline entrypoint
├── run_optimization_pipeline.py    # [CLI] TensorRT compile benchmarks entrypoint
└── README.md                       # High-level overview (this file)
```

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Unit Tests
Execute the pytest suite to verify all modules and dimensions:
```bash
python -m pytest
```

### 3. Run the Specialized Pipelines
* **Generate Synthetic Scans & Documents**:
  ```bash
  python run_synthetic_pipeline.py
  ```
* **Audit Failures and Extract Explainability Saliency**:
  ```bash
  python run_explainability_pipeline.py
  ```
* **Log Runs to MLflow, W&B, and TensorBoard**:
  ```bash
  python run_tracking_pipeline.py
  ```
* **Run Distributed and Compile TensorRT Engines**:
  ```bash
  python run_optimization_pipeline.py
  ```

### 4. Deploying the REST API
Start the FastAPI server:
```bash
python -m uvicorn src.production.app:app --host 127.0.0.1 --port 8000
```
* **Health Check**: `GET http://127.0.0.1:8000/health`
* **Inference Endpoint**: `POST http://127.0.0.1:8000/predict`
* **Synthetic Generator**: `POST http://127.0.0.1:8000/synthetic/generate`
* **Explainability Audit**: `POST http://127.0.0.1:8000/explainability/analyze`
* **Performance Benchmark**: `POST http://127.0.0.1:8000/optimization/benchmark`

