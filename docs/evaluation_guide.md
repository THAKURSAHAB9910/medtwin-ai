# MedTwin AI: Evaluation Guide

## 1. Automated Diagnostic Verification
MedTwin AI metrics evaluation suite checks model reliability across medical domains. Run the complete pytest verification pipeline:

```bash
python -m pytest
```

---

## 2. Metric Specifications

### 2.1 Medical OCR Engine
Character Accuracy (CA) is calculated using edit distance checks:
$$\text{CA} = 1.0 - \frac{\text{LevenshteinDistance}(Y_{\text{pred}}, Y_{\text{truth}})}{\max(|Y_{\text{pred}}|, |Y_{\text{truth}}|)}$$
* Target threshold: $>95\%$ for typed documents, $>90\%$ for handwritten scans.

### 2.2 Segmentations and Localization
* **Intersection over Union (IoU)**: Evaluates bounding box overlaps:
  $$\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{truth}}|}{|M_{\text{pred}} \cup M_{\text{truth}}|}$$
* **Centroid distance**: Pinpoints diagnostic center coordinates.

### 2.3 RAG Faithfulness
Evaluates generated answer support relative to context citations using clinical entity matches:
$$\text{Faithfulness} = \frac{\text{Supported NER Medical Entities}}{\text{Total Extracted Medical Entities}}$$
* Target threshold: $>85\%$ score.

---

## 3. Launching Benchmarks Run
To benchmark model processing latencies and accuracy thresholds locally:
```bash
python run_optimization_pipeline.py
python run_synthetic_pipeline.py
python run_explainability_pipeline.py
```
Outputs visual performance plots inside the `docs/` directory.
