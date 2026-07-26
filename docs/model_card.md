# MedTwin AI: Model Card

## 1. Model Details
* **Developer**: MedTwin AI Research Group.
* **Architecture**: Multi-modal fusion model utilizing custom convolutional backbones for visual scans and transformer tokenizers for clinical text, wrapped with parameter-efficient fine-tuning (PEFT/LoRA).
* **Version**: v1.3 (optimized for TensorRT execution).
* **License**: Apache 2.0.

## 2. Intended Use
* **Primary Objective**: Clinical decision support. Assists clinicians by isolating scan abnormalities, transcribing prescription documents, compiling patient history timelines, and flagging drug safety contraindications.
* **Out-of-Scope**: Automated diagnostics without physician validation.

---

## 3. Optimization & Resource footprint
* **Base Parameters**: 140M parameters.
* **Fine-Tuning Hook (LoRA)**: Rank $r = 16$, Target modules `[q_proj, v_proj]`, Alpha $\alpha = 32$.
* **Precision Formats**: FP32 (Baseline), FP16 (Mixed-precision training), INT8 (Dynamic dynamic quantization for inference).
* **Peak GPU memory**: 2.8 GB (using TensorRT engine configuration).

---

## 4. Calibration & Saliency Targets
* **Uncertainty Calibration**: Uses temperature scaling:
  $$\hat{q}_i = \sigma(z_i / T)$$
  where Temperature parameter $T = 1.15$, yielding calibrated probabilities matching true model precision rates.
* **Explainability Saliency Hooks**:
  * Grad-CAM: Target Layer `backbone.layer4.conv2`.
  * SHAP: Calculates game-theoretic values over vitals parameters (Oxygen saturation, Temperature, WBC).
