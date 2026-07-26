import numpy as np
from PIL import Image, ImageFilter
from typing import Dict, Any, List

class SyntheticMedicalDataEngine:
    """
    Synthesizes degraded imaging scans (low-quality, blurred, noisy) and 
    clinical text documents (prescriptions, summaries, histories) including rare diseases.
    """
    def __init__(self):
        self.rare_diseases = [
            "alkaptonuria", 
            "huntingtons_disease", 
            "fibrodysplasia_ossificans_progressiva",
            "glioblastoma"
        ]

    def generate_image(self, modality: str, condition: str, quality: str = "high") -> Image.Image:
        """
        Generates synthetic medical scan images with optional visual degradations.
        """
        # Create standard grey baseline canvas
        arr = np.ones((256, 256, 3), dtype=np.uint8) * 128
        
        # Draw a mock pathology shape in the center
        center_val = 200 if condition.lower() in self.rare_diseases else 50
        arr[90:160, 90:160, :] = center_val
        
        img = Image.fromarray(arr)
        
        # Apply visual degradations
        q = quality.lower().strip()
        if q == "blurred":
            img = img.filter(ImageFilter.GaussianBlur(radius=4.5))
        elif q == "noisy":
            # Add Gaussian noise
            np_img = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 25, np_img.shape)
            noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy_img)
        elif q == "low-quality":
            # Subsample and rescale back to induce pixelation artifacts
            small = img.resize((32, 32), Image.Resampling.NEAREST)
            img = small.resize((256, 256), Image.Resampling.NEAREST)
            
        return img

    def generate_prescription(self, condition: str) -> str:
        """
        Generates a synthetic drug prescription matching the condition.
        """
        cond = condition.lower().strip().replace(" ", "_")
        
        prescriptions = {
            "alkaptonuria": "Rx: Nitisinone 10mg PO daily. High-dose Vitamin C 500mg PO twice daily. Avoid high-protein diets.",
            "huntingtons_disease": "Rx: Tetrabenazine 12.5mg PO daily. Titrate by 12.5mg weekly as tolerated for chorea control.",
            "fibrodysplasia_ossificans_progressiva": "Rx: Palovarotene 5mg PO daily. Take with food. Discontinue if flare-ups subside.",
            "glioblastoma": "Rx: Temozolomide 150mg/m2 PO daily for 5 days of a 28-day cycle. Compulsory anti-emetic premedication.",
            "pneumonia": "Rx: Ceftriaxone 1g IV daily plus Azithromycin 500mg PO daily for 7 days."
        }
        
        return prescriptions.get(cond, "Rx: Supportive care. Hydration therapy. Routine symptom management.")

    def generate_discharge_summary(self, condition: str) -> str:
        """
        Generates a synthetic discharge note outlining the clinical course.
        """
        cond = condition.upper().strip()
        return (
            f"DISCHARGE NARRATIVE:\n"
            f"Patient diagnosed with {cond}. Clinical course was characterized by specialized evaluations.\n"
            f"Active symptoms resolved post-therapy. Follow-up scheduled with multidisciplinary clinic in 14 days."
        )

    def generate_lab_report(self, condition: str) -> str:
        """
        Generates synthetic lab results panel.
        """
        cond = condition.lower().strip().replace(" ", "_")
        
        if cond == "alkaptonuria":
            return "URINALYSIS PANEL: Homogentisic Acid: 8.2 g/24h (High). Urine darkens on standing: Positive."
        elif cond == "glioblastoma":
            return "HISTOLOGY AUDIT: IDH-1 mutation status: Wildtype. Methylation of MGMT promoter: Negative."
        else:
            return "METABOLIC PANEL: Glucose: 98 mg/dL. WBC: 6500 /uL. Creatinine: 0.9 mg/dL. Hemoglobin: 14.2 g/dL."

    def generate_patient_history(self, condition: str) -> str:
        """
        Generates a synthetic patient medical history narrative.
        """
        cond = condition.lower().strip().replace(" ", "_")
        
        histories = {
            "alkaptonuria": "Patient history: Early-onset osteoarthritis starting in joint joints. Noticed dark stains on underwear since infancy.",
            "huntingtons_disease": "Patient history: Family history positive for chorea. Progressive cognitive and motor decline reported by spouse.",
            "fibrodysplasia_ossificans_progressiva": "Patient history: Bilateral malformed great toes noted at birth. Progressive heterotopic ossification following minor muscle trauma.",
            "glioblastoma": "Patient history: Subacute onset of progressive headache, nausea, and expressive aphasia. MRI confirms ring-enhancing mass."
        }
        
        return histories.get(cond, "Patient history: Normal baseline metrics. Non-contributory clinical histories.")
