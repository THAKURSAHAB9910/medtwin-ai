from PIL import Image
from typing import Dict, Any, List

from src.nlp.entity_extractor import ClinicalNER

class MedicalOCRPipeline:
    """
    Unified Medical OCR pipeline supporting prescriptions, lab sheets,
    discharge summaries, insurance sheets, and handwritten doctors notes.
    """
    def __init__(self, ner: ClinicalNER):
        self.ner = ner
        self.doc_types = [
            "prescription", "lab report", "discharge summary", 
            "insurance document", "referral letter", 
            "handwritten prescription", "medical certificate"
        ]

    def _sanitize_doc_type(self, doc_type: str) -> str:
        dt = doc_type.lower().strip()
        if dt not in self.doc_types:
            return "prescription"
        return dt

    def extract_text_and_entities(self, image: Image.Image, doc_type: str) -> Dict[str, Any]:
        """
        Runs OCR simulation and extracts named clinical entities.
        """
        dt = self._sanitize_doc_type(doc_type)
        
        # Document template texts
        templates = {
            "prescription": "Rx: Metformin 500mg PO twice daily. Refills: 3. Signed, Dr. Smith.",
            "lab report": "HEMATOLOGY PANEL: WBC count: 14500 /uL (High). Glucose: 165 mg/dL. HbA1c: 7.8%. Diagnosis: Diabetes mellitus.",
            "discharge summary": "Discharge diagnosis: Community-Acquired Pneumonia. Active medications: Ceftriaxone 1g IV daily.",
            "insurance document": "CLAIM PROFILE: Policy #9923812. Diagnosis code: Pneumonia. Approved treatment class: IV Antibiotics.",
            "referral letter": "Referral to Pulmonology for evaluation of dyspnea and persistent cough. Suspect active pneumonia.",
            "handwritten prescription": "Ceftriaxone 1g IV daily x 7 days. Signed, Dr. Jones.",
            "medical certificate": "CERTIFICATE OF ILLNESS: Patient diagnosed with Pneumonia. Excused from duty for 5 days."
        }
        
        raw_text = templates[dt]
        entities = self.ner.extract_entities(raw_text)
        
        return {
            "document_type": dt,
            "ocr_text": raw_text,
            "extracted_medications": entities["medications"],
            "extracted_dosages": entities["dosages"],
            "extracted_diseases": entities["diseases"],
            "extracted_symptoms": entities["symptoms"]
        }
