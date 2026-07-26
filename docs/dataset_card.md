# MedTwin AI: Dataset Card

## 1. Dataset Summary
MedTwin AI utilizes a heterogeneous multi-modal patient cohort dataset linking medical scans, tabular vitals, clinical reports, and reference guidelines. This cohort supports diagnostic classification, anomaly localization, document OCR extraction, and digital twin patient state progression modelling.

## 2. Cohort Cohort Statistics
* **Total Patients Profiled**: 15,000 cases.
* **Scan Modalities**:
  * Radiographs (Chest X-Rays): 5,000 cases (Pneumonia, Healthy).
  * Brain MRI Scans: 3,000 cases (Glioblastoma, Huntington's).
  * Retinal Scans: 2,500 cases (Diabetic Retinopathy).
  * Skin Lesions (Histopathology): 2,000 cases (Dermatology, Melanoma).
  * Tabular Patient Histories: 2,500 cases (Vitals, ECG sequences).
* **Rare Diseases (Orphan cohort)**: Includes Alkaptonuria, Huntington's disease, Fibrodysplasia Ossificans Progressiva (FOP), and Glioblastoma.

## 3. Data Format & Schema
### 3.1 Visual Scans
* **Format**: PNG, RGB.
* **Resolution**: 256x256 or 384x384.
* **Annotation**: Bounding boxes `[xmin, ymin, xmax, ymax]` for lesion localizations.

### 3.2 Tabular Vitals
* `temperature`: Body temp (°F). Normal: [97.5 - 99.0], Pathological: >100.4.
* `wbc_count`: White Blood Cell Count (cells/µL). Normal: [4,500 - 11,000], Leucocytosis: >11,000.
* `oxygen_saturation`: SpO2 (%) range. Normal: >95%, Hypoxia: <94%.

---

## 4. Synthetic Degradation Parameters
To stress-test diagnostic robustness, clean scans are augmented using:
* **Gaussian Blur**: Kernel size $\sigma \in [1.0, 3.0]$.
* **Gaussian Noise**: Mean $\mu = 0$, Variance $\sigma^2 \in [0.01, 0.05]$.
* **Pixelation**: Downsampled by a factor of 4.
