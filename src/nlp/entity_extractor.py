import re
from typing import Dict, List, Any

class ClinicalNER:
    """
    Extracts medical Named Entities (medications, symptoms, diseases, dosage)
    and evaluates drug-to-drug clinical interactions.
    """
    def __init__(self):
        # Entity matching dictionaries
        self.med_patterns = {
            "Metformin": r"\b(metformin|glucophage)\b",
            "Ceftriaxone": r"\b(ceftriaxone|rocephin)\b",
            "Azithromycin": r"\b(azithromycin|zithromax)\b",
            "Aspirin": r"\b(aspirin|asa)\b",
            "Lisinopril": r"\b(lisinopril|prinivil)\b",
            "Spironolactone": r"\b(spironolactone|aldactone)\b",
            "Ibuprofen": r"\b(ibuprofen|advil)\b"
        }
        
        self.symptom_patterns = {
            "cough": r"\b(cough|coughing)\b",
            "fever": r"\b(fever|pyrexia|chills)\b",
            "dyspnea": r"\b(dyspnea|shortness of breath|sob)\b",
            "chest pain": r"\b(chest pain|angina)\b",
            "polyuria": r"\b(polyuria|frequent urination)\b"
        }
        
        self.disease_patterns = {
            "Pneumonia": r"\b(pneumonia|pulmonary infection)\b",
            "Diabetes": r"\b(diabetes|type 2 diabetes|dm2)\b",
            "Hypertension": r"\b(hypertension|high blood pressure|htn)\b",
            "Stroke": r"\b(stroke|cva|transient ischemic attack)\b"
        }
        
        self.dosage_pattern = r"\b(\d+\s*(mg|g|mcg|ml))\b"

        # Drug-drug interaction database
        self.interactions = [
            {
                "drugs": ["Metformin", "Contrast Dye"],
                "severity": "High",
                "warning": "Metformin + Iodinated Contrast Agents poses severe risk of lactic acidosis. Suspend Metformin 48 hours post-scan."
            },
            {
                "drugs": ["Aspirin", "Ibuprofen"],
                "severity": "Moderate",
                "warning": "Ibuprofen may limit the cardioprotective antiplatelet effect of low-dose Aspirin."
            },
            {
                "drugs": ["Lisinopril", "Spironolactone"],
                "severity": "High",
                "warning": "Co-administration of ACE inhibitors (Lisinopril) and potassium-sparing diuretics (Spironolactone) increases risk of hyperkalemia."
            }
        ]

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Scans medical text and extracts matched entity terms.
        """
        text_l = text.lower()
        extracted = {
            "medications": [],
            "symptoms": [],
            "diseases": [],
            "dosages": []
        }
        
        # 1. Match medications
        for med, pattern in self.med_patterns.items():
            if re.search(pattern, text_l):
                extracted["medications"].append(med)
                
        # Handle implicit mock contrast dye match for scan requests
        if "contrast" in text_l or "dye" in text_l:
            extracted["medications"].append("Contrast Dye")

        # 2. Match symptoms
        for sym, pattern in self.symptom_patterns.items():
            if re.search(pattern, text_l):
                extracted["symptoms"].append(sym)

        # 3. Match diseases
        for dis, pattern in self.disease_patterns.items():
            if re.search(pattern, text_l):
                extracted["diseases"].append(dis)

        # 4. Match dosages
        dosages = re.findall(self.dosage_pattern, text_l)
        extracted["dosages"] = [d[0] for d in dosages]

        return extracted

    def check_drug_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """
        Compares list of active medications against interaction database.
        """
        warnings = []
        med_set = {m.lower() for m in medications}
        
        for inter in self.interactions:
            match_count = 0
            for d in inter["drugs"]:
                if d.lower() in med_set:
                    match_count += 1
            if match_count == len(inter["drugs"]):
                warnings.append({
                    "severity": inter["severity"],
                    "warning": inter["warning"],
                    "drugs": inter["drugs"]
                })
        return warnings
