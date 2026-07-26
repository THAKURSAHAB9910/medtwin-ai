import numpy as np
from typing import Dict, Any, List
from src.agents.state import SpecialistOpinion

class SpecialistAgent:
    """
    Base class for all Specialized Medical AI Agents.
    Simulates diagnostic logic based on specialist priorities.
    """
    def __init__(self, role: str):
        self.role = role

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        """
        Analyzes multimodal patient record. Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement analyze method.")


class RadiologistAgent(SpecialistAgent):
    """
    Radiology Specialist: Focuses on chest radiographs and visual pathology/lesions.
    """
    def __init__(self):
        super().__init__("Radiologist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        note = patient_data.get("clinical_note", "").lower()
        image_present = "image" in patient_data
        
        # Check visual flags or note indicators for lung consolidation
        is_consolidation = "consolidation" in note or "opacity" in note or "haziness" in note or "infiltration" in note
        
        if is_consolidation or (image_present and "pneumonia" in note):
            diagnosis = "Pneumonia"
            confidence = 0.90
            evidence = "Chest radiograph demonstrates marked focal lung opacity and consolidation in the lower lobes, indicating active alveolar infiltration."
            recommended_tests = ["High-Resolution CT Thorax"]
            treatment_suggestions = ["Respiratory support and monitoring"]
        else:
            diagnosis = "Normal"
            confidence = 0.85
            evidence = "No active lung consolidation or focal pleural effusion identified on chest radiograph. Lung cavities appear clear."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class CardiologistAgent(SpecialistAgent):
    """
    Cardiology Specialist: Focuses on heart rate, chest pain, and ECG vitals.
    """
    def __init__(self):
        super().__init__("Cardiologist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        vitals = patient_data.get("lab_metrics", {})
        hr = vitals.get("heart_rate", 75.0)
        note = patient_data.get("clinical_note", "").lower()
        
        is_cardiac = "chest pain" in note or "angina" in note or "heart" in note
        
        if hr > 100.0 and is_cardiac:
            diagnosis = "Myocardial Infarction"
            confidence = 0.85
            evidence = f"Tachycardia of {hr} bpm accompanied by severe chest pain. Indication of potential myocardial ischemia."
            recommended_tests = ["12-Lead ECG", "Cardiac Troponin I/T Panel", "Echocardiogram"]
            treatment_suggestions = ["Immediate Antiplatelet loading (Aspirin 325mg)", "Sublingual Nitroglycerin"]
        elif hr > 95.0 and "fever" in note:
            # Secondary tachycardia due to pulmonary infection
            diagnosis = "Pneumonia"
            confidence = 0.70
            evidence = f"Mild sinus tachycardia of {hr} bpm likely secondary to febrile infectious state, rather than primary cardiac pathology."
            recommended_tests = ["Serial ECG monitoring"]
            treatment_suggestions = ["Hydration control"]
        else:
            diagnosis = "Normal"
            confidence = 0.90
            evidence = f"Normal cardiac heart rate of {hr} bpm. No reports of chest pain or ischemic ECG anomalies."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class NeurologistAgent(SpecialistAgent):
    """
    Neurology Specialist: Focuses on mental status, stroke vitals, and brain signals.
    """
    def __init__(self):
        super().__init__("Neurologist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        note = patient_data.get("clinical_note", "").lower()
        history = patient_data.get("patient_history", "").lower()
        
        is_neurological = "altered mental status" in note or "speech difficulty" in note or "weakness" in note
        
        if is_neurological:
            diagnosis = "Acute Stroke"
            confidence = 0.88
            evidence = "Patient presents with acute focal neurological deficits: altered mental status or speech difficulties. Suspicious of ischemic stroke."
            recommended_tests = ["Non-Contrast Head CT Scan", "Brain MRI"]
            treatment_suggestions = ["Evaluate for IV Alteplase within thrombolytic window"]
        else:
            diagnosis = "Normal"
            confidence = 0.95
            evidence = "Patient is alert and oriented. No focal neurological deficits or speech abnormalities documented."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class PathologistAgent(SpecialistAgent):
    """
    Pathology Specialist: Focuses on lab blood panels and WBC counts.
    """
    def __init__(self):
        super().__init__("Pathologist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        vitals = patient_data.get("lab_metrics", {})
        wbc = vitals.get("wbc_count", 7500.0)
        
        if wbc > 12000.0:
            diagnosis = "Pneumonia"
            confidence = 0.82
            evidence = f"Marked leukocytosis with WBC count of {wbc}/uL, indicating a severe systemic immune response consistent with bacterial infection."
            recommended_tests = ["Sputum Gram Stain & Culture", "Blood Cultures x2"]
            treatment_suggestions = ["Broad-spectrum empiric antibiotics"]
        else:
            diagnosis = "Normal"
            confidence = 0.88
            evidence = f"WBC count within normal physiological limits ({wbc}/uL). No active markers of systemic leukocytosis."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class PharmacistAgent(SpecialistAgent):
    """
    Pharmacology Specialist: Reviews prescription records and suggests drug regimens.
    """
    def __init__(self):
        super().__init__("Pharmacist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        note = patient_data.get("clinical_note", "").lower()
        vitals = patient_data.get("lab_metrics", {})
        temp = vitals.get("temperature", 98.6)
        
        # Check if patient presents with systemic infection parameters
        if temp > 100.4 or "infection" in note or "pneumonia" in note:
            diagnosis = "Pneumonia"
            confidence = 0.75
            evidence = f"Fever of {temp}F indicating acute infectious profile. Requires antibiotic intervention."
            recommended_tests = ["Renal function panel (CrCl) to dose antibiotics"]
            treatment_suggestions = ["Empiric Antibiotics: Ceftriaxone 1-2g IV daily + Azithromycin 500mg IV/PO daily"]
        else:
            diagnosis = "Normal"
            confidence = 0.90
            evidence = "No pharmacological indicators or active toxic/infectious signs noted. Current drug profile is stable."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class NutritionistAgent(SpecialistAgent):
    """
    Nutrition Specialist: Recommends supportive nutritional and hydration care.
    """
    def __init__(self):
        super().__init__("Nutritionist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        vitals = patient_data.get("lab_metrics", {})
        temp = vitals.get("temperature", 98.6)
        
        if temp > 100.4:
            diagnosis = "Pneumonia"
            confidence = 0.65
            evidence = f"Elevated temperature ({temp}F) increases metabolic rate and insensible fluid loss, indicating hydration deficits."
            recommended_tests = ["Basic Metabolic Panel (Electrolytes, BUN)"]
            treatment_suggestions = ["Intravenous hydration therapy", "High-calorie clinical nutrition shake supplements"]
        else:
            diagnosis = "Normal"
            confidence = 0.85
            evidence = "Patient metabolic demand is baseline. Vitals do not indicate immediate nutritional deficits."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class MedicalResearcherAgent(SpecialistAgent):
    """
    Medical Researcher: Cross-references clinical presentations with SOTA research classifications.
    """
    def __init__(self):
        super().__init__("Medical Researcher")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        note = patient_data.get("clinical_note", "").lower()
        
        if "cough" in note and "fever" in note:
            diagnosis = "Pneumonia"
            confidence = 0.80
            evidence = "Clinical signs (cough, fever) align with established epidemiological diagnostics for community-acquired pneumonia."
            recommended_tests = ["Polymerase Chain Reaction (PCR) Respiratory Viral Panel"]
            treatment_suggestions = ["Antimicrobial stewardship compliance review"]
        else:
            diagnosis = "Normal"
            confidence = 0.88
            evidence = "Clinical presentation does not match active diagnostic cohorts for acute respiratory or cardiovascular disease."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )


class DermatologistAgent(SpecialistAgent):
    """
    Dermatologist: Focuses on skin lesions, rashes, and cutaneous abnormalities.
    """
    def __init__(self):
        super().__init__("Dermatologist")

    def analyze(self, patient_data: Dict[str, Any]) -> SpecialistOpinion:
        note = patient_data.get("clinical_note", "").lower()
        
        is_skin = "rash" in note or "lesion" in note or "hives" in note
        
        if is_skin:
            diagnosis = "Cutaneous Infection"
            confidence = 0.85
            evidence = "Cutaneous presentation shows rash or active localized skin lesions."
            recommended_tests = ["Skin Lesion Biopsy", "Wound Culture"]
            treatment_suggestions = ["Topical corticosteroids", "Antifungals"]
        else:
            diagnosis = "Normal"
            confidence = 0.95
            evidence = "Skin and integumentary system are intact. No active rashes or lesions observed."
            recommended_tests = []
            treatment_suggestions = []

        return SpecialistOpinion(
            role=self.role,
            diagnosis=diagnosis,
            confidence=confidence,
            evidence=evidence,
            recommended_tests=recommended_tests,
            treatment_suggestions=treatment_suggestions
        )
