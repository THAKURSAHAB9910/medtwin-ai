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
