# MedTwin AI: Technical System Architecture Report

This report outlines the technical specifications of **MedTwin AI**, an uncertainty-calibrated, self-correcting clinical foundation model extending into a multi-agent Clinical Consensus Board and a Patient Digital Twin.

---

## 1. Tri-Modal Feature Extraction & Projection

MedTwin AI ingests three heterogeneous patient clinical modalities:
1. **Medical Scan ($I \in \mathbb{R}^{H \times W \times 3}$)**: Processed by `MedicalVisualBranch` utilizing a ResNet-18 feature encoder. It outputs local feature maps $F_{\text{vis}} \in \mathbb{R}^{C \times H_{\text{feat}} \times W_{\text{feat}}}$ ($C=512$, $H_{\text{feat}}=12, W_{\text{feat}}=12$) and reconstructs dense lung lesion maps using skip-connected upsampling:
   $$\hat{M} = \text{Decoder}(F_{\text{vis}})$$
2. **Clinical Note ($N$)**: Tokenized and processed by a clinical language model (`ClinicalVLM`), yielding token embeddings $F_{\text{txt}} \in \mathbb{R}^{L \times D}$ ($D=256$, sequence length $L=64$).
3. **Structured Lab Metrics & Vitals ($V \in \mathbb{R}^{5}$)**: Comprises `[age, temperature, heart_rate, wbc_count, oxygen_saturation]`. Processed by a BatchNormalization-MLP network (`StructuredLabEncoder`) to generate a clinical vitals embedding token $F_{\text{lab}} \in \mathbb{R}^{1 \times D}$.

---

## 2. Tri-Modal Gated Cross-Attention Fusion

We align the different modalities in a unified clinical reasoning layer:
1. **Context Sequence ($K, V$)**: We concatenate the textual notes and structured vitals along the token sequence axis:
   $$X_{\text{context}} = [ F_{\text{txt}} \;\|\; F_{\text{lab}} ] \;\in\; \mathbb{R}^{(L+1) \times D}$$
2. **Visual Queries ($Q$)**: The visual features are flattened and projected to the shared dimension:
   $$X_{\text{query}} = \text{Flatten}(F_{\text{vis}}) \cdot W_p \;\in\; \mathbb{R}^{(H_{\text{feat}} W_{\text{feat}}) \times D}$$
3. **Multi-Head Cross-Attention**: Visual pixels query the diagnostic terms in notes and lab indices:
   $$X_{\text{attn}} = \text{Softmax}\left(\frac{Q K^T}{\sqrt{D}}\right) V$$
4. **Adaptive Gating**: Blends the raw visual representation with attention-fused patient contexts using an input-dependent sigmoid gating factor:
   $$\lambda = \sigma\left(W_{\text{gate}} \cdot \text{pool}(F_{\text{vis}}) + b_{\text{gate}}\right) \in [0, 1]$$
   $$X_{\text{fused}} = (1 - \lambda) \cdot X_{\text{query}} + \lambda \cdot X_{\text{attn}}$$
5. **Final Diagnostic Logits**: Fused tokens are pooled and classified:
   $$z = \text{Classifier}\left(\text{mean}(X_{\text{fused}})\right) \;\in\; \mathbb{R}$$

---

## 3. Clinical Consensus Engine (Multi-Agent Architecture)

To emulate clinical medical boards, MedTwin AI deploys a Multi-Agent system coordinated via a LangGraph-style clinical state:
1. **Specialist Agents**: 8 autonomous specialist units evaluate the shared state:
   - **Radiologist**: Evaluates chest radiographs for visual opacities and lesions.
   - **Cardiologist**: Focuses on heart rate, chest pain symptoms, and ischemic ECG runs.
   - **Neurologist**: Analyzes focal deficits (speech, motor, altered mental state).
   - **Pathologist**: Reviews laboratory blood metrics (leukocytosis boundaries).
   - **Pharmacist**: Audits prescription dosing, contraindications, and therapeutic regimens.
   - **Nutritionist**: Formulates fluid intake and metabolic/dietary requirements.
   - **Medical Researcher**: Matches diagnostic cohorts with medical literature trends.
   - **Dermatologist**: Examines cutaneous rashes and integumentary anomalies.
2. **LangGraph Clinical State**: A shared Pydantic object storing patient records, specialist outputs (Diagnosis, Confidence, Evidence, Recommended Tests/Treatments), and logs coordinator progress.
3. **Coordinator Agent**: Manages parallel execution, computes majority-voting consensus, determines disagreements, and queries the **Medical Knowledge Graph** (ICD-10 maps) and **Medical RAG** guideline database.

---

## 4. Patient Digital Twin (Longitudinal Simulation Engine)

The Patient Digital Twin models dynamic physiological state transitions over time under user-specified "What-If" clinical interventions:
1. **State Vector ($S_t \in \mathbb{R}^6$)**: Represents clinical parameters:
   $$S_t = [\text{weight}, \text{blood\_pressure}, \text{blood\_glucose}, \text{heart\_rate}, \text{wbc\_count}, \text{oxygen\_saturation}]$$
2. **Action Vector ($A_t \in \mathbb{R}^5$)**: Represents intervention targets:
   $$A_t = [\Delta\text{weight}, \text{metformin\_mg}, \text{beta\_blocker\_mg}, \text{exercise\_hours}, \text{sodium\_intake\_g}]$$
3. **Bayesian State-Space Transition**: Represents transitions $S_{t+1} = S_t + f(S_t, A_t) + \epsilon$. The model $f$ is trained using PyTorch with active Monte Carlo (MC) Dropout to estimate epistemic uncertainty:
   $$\sigma^2 = \text{Var}_{\text{MC}}(S_{t+1})$$
   We run $M=50$ forward passes to construct the 95% Bayesian credible intervals ($\text{Mean} \pm 1.96 \cdot \text{Std}$).
4. **Physiological Rule Layer**: Overlays hard constraints from the Knowledge Graph:
   - Metformin: Reduces glucose recursively.
   - Weight loss: Drops systolic blood pressure by 1 mmHg per kg lost.
5. **Citations & Guidelines**: The RAG subsystem retrieves official American Diabetes Association (ADA Standards of Care) and American Heart Association (AHA Hypertension guidelines) details based on the final forecasted states.

---

## 5. Medical Vision Language Model (VLM) Research Module

To support multimodal clinical grounding and document OCR processing, MedTwin AI integrates a Unified VLM registry:
1. **Model Specifications**:
   - **Qwen2-VL**: High-resolution vision-language model processing spatial visual frames and complex multi-modal reports.
   - **BioMedCLIP**: Language-image contrastive model matching visual embeddings $E_v$ to clinical text label embeddings $E_t$:
     $$\text{Similarity} = \frac{E_v \cdot E_t}{\|E_v\| \|E_t\|}$$
   - **Florence-2**: Localized vision model returning bounding boxes $B = [x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ for visual pathology grounding.
   - **LayoutLMv3**: Multi-modal layout-aware OCR classifier combining visual, text, and 2D spatial position embeddings.
2. **Unified Inference Interface**: Exposes a common API to switch models dynamically and extract structured confidence indicators.

---

## 6. Clinical Fine-Tuning & Adapter Pipeline (PEFT)

To optimize biomedical foundation models on patient note vocabularies without unfreezing raw layers, MedTwin AI implements Parameter-Efficient Fine-Tuning (PEFT):
1. **LoRA (Low-Rank Adaptation)**: Parametrizes weight updates $\Delta W$ by factorizing it into two low-rank matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ (where rank $r \ll \min(d, k)$):
   $$W_{\text{updated}} = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (A \cdot B)$$
   During Supervised Fine-Tuning (SFT), base weights $W_0$ remain frozen ($\nabla_{W_0} \mathcal{L} = 0$), drastically reducing optimizer VRAM requirements.
2. **QLoRA (Quantized Low-Rank Adaptation)**: Quantizes the base weights $W_0$ to 4-bit NormalFloat (NF4) representations, loading double-quantized scaling factors. During backpropagation, the gradients are calculated exclusively on active high-precision 16-bit LoRA adapter parameters:
   $$Y = \text{dequantize}(W_{\text{NF4}}) \cdot X + \frac{\alpha}{r} (A \cdot B) \cdot X$$
3. **Supervised Fine-Tuning Loss**: Calculated using cross-entropy over target labels while masking user prompt prompts:
   $$\mathcal{L}_{\text{SFT}} = -\sum_{i \in T} \log P(y_i \mid y_{<i}, X_{\text{prompt}})$$
   where prompt tokens $i \notin T$ are masked out (set to $-100$).

---

## 7. Medical NLP & Clinical RAG Pipeline

To optimize semantic search, patient timelines, and QA evaluation over clinical documents, MedTwin AI implements a Medical NLP pipeline:
1. **Cosine Similarity Retrieval**: Let query embedding be $q$ and document embedding be $d_i$, both unit-normalized. Cosine similarity calculates retrieval rankings:
   $$\text{Sim}(q, d_i) = q \cdot d_i = \sum_{j=1}^{D} q_j d_{i, j}$$
2. **Faithfulness & Hallucination Metrics**: Faithfulness measures context alignment, while Hallucination represents unsupported claims:
   $$\text{Faithfulness} = \frac{|E_{\text{answer}} \cap (E_{\text{context}} \cup E_{\text{query}})|}{|E_{\text{answer}}|}$$
   $$\text{Hallucination Rate} = 1.0 - \text{Faithfulness}$$
   where $E_{\text{answer}}$ represents the Named Entities (medications, symptoms, diseases) extracted from the generated answer.
3. **Timeline Compilation**: Orders chronological clinical events by mapping datetime timestamps:
   $$t_0 < t_1 < \dots < t_N$$

---

## 8. Multimodal Clinical Intelligence & Joint Reasoning

To unify diagnostics across visual, textual, tabular, and genomic data inputs, MedTwin AI implements a joint clinical reasoning module:
1. **Multimodal State Vector**: Ingests patient note tokens $T$, vital indices $V$, chronological narratives $C$, and genomic mutation binaries $G$:
   $$S = \{T, V, C, G\}$$
2. **Joint Diagnostic Likelihood**: Determines primary diagnosis $D_k$ using aligned visual opacities and vital thresholds:
   $$P(D_k \mid S) = \sigma(\text{Linear}(E_v) + \beta_{\text{temp}} (T_{\text{vital}} - 98.6) + \gamma_g G_{\text{mut}})$$
   where $E_v$ represents the visual VLM radiograph features, and $G_{\text{mut}}$ matches active mutations (e.g. EGFR).
3. **Evidence Explainability**: Automatically traces active pathways (SIRS indicators, hypoxemia) to produce structural justifications of diagnostic confidence scores.

---

## 9. Multi-Agent Board Consensus & Uncertainty Mechanics

To evaluate diagnosis alignment across isolated medical specialities, MedTwin AI coordinates a Multi-Agent board consisting of $N=7$ specialists:
1. **Consensus Vote**: Each agent $A_i$ produces an independent opinion diagnostic label $D_i$ with confidence $c_i$. The board resolves consensus via plurality voting:
   $$D_{\text{consensus}} = \arg\max_D \sum_{i=1}^{N} \mathbb{I}(D_i = D)$$
   If there is a tie, it selects the diagnosis with the highest cumulative confidence $\sum c_i$.
2. **Board Agreement Rate**: Computes the ratio of specialists conforming to the majority:
   $$\text{Agreement Rate} = \frac{\sum_{i=1}^{N} \mathbb{I}(D_i = D_{\text{consensus}})}{N}$$
3. **Consensus Uncertainty**: Quantifies conflict entropy or misalignment bounds among board opinions:
   $$\text{Uncertainty} = 1.0 - \text{Agreement Rate}$$

---

## 10. Medical Image Research & Vision Formulations

To process heterogeneous imaging modalities (X-ray, MRI, CT, Retinal, Skin, Histo, Micro, Ultrasound, Wound) across diverse tasks (Segmentation, Classification, Detection, Localization, Severity), MedTwin AI implements specialized vision metrics:
1. **Intersection over Union (IoU)**: Evaluates pixel-level lesion segmentation boundaries relative to ground truth masks:
   $$\text{IoU} = \frac{|M_{\text{pred}} \cap M_{\text{truth}}|}{|M_{\text{pred}} \cup M_{\text{truth}}|}$$
2. **Mean Average Precision (mAP)**: Evaluates bounding box coordinates for anomaly detection at varied IoU thresholds.
3. **Centroid Localization**: Maps bounding box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ to anomaly center coordinates:
   $$C = \left[ \frac{x_{\text{min}} + x_{\text{max}}}{2}, \frac{y_{\text{min}} + y_{\text{max}}}{2} \right]$$
