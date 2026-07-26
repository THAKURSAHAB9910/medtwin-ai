from typing import List, Dict, Tuple, Set

class MedicalKnowledgeGraph:
    """
    In-memory clinical Knowledge Graph representing medical entity relations.
    Models medical knowledge as Subject-Predicate-Object triples, mapping diseases
    to symptoms, ICD-10 codes, lab markers, and recommended clinical checks.
    """
    def __init__(self):
        self.triples: Set[Tuple[str, str, str]] = set()
        self._initialize_graph()

    def _initialize_graph(self):
        # Triples mapping Pneumonia (Respiratory)
        self.triples.update([
            ("Pneumonia", "has_symptom", "cough"),
            ("Pneumonia", "has_symptom", "fever"),
            ("Pneumonia", "has_symptom", "dyspnea"),
            ("Pneumonia", "has_symptom", "crackles"),
            ("Pneumonia", "has_lab_marker", "wbc_count"),
            ("Pneumonia", "has_lab_marker", "temperature"),
            ("Pneumonia", "has_lab_marker", "oxygen_saturation"),
            ("Pneumonia", "has_icd10_code", "J18.9"),
            ("Pneumonia", "requires_test", "Chest Radiograph"),
            ("Pneumonia", "requires_test", "Sputum Culture"),
            ("Pneumonia", "has_treatment_class", "Antibiotics")
        ])

        # Triples mapping Myocardial Infarction (Cardiology)
        self.triples.update([
            ("Myocardial Infarction", "has_symptom", "chest pain"),
            ("Myocardial Infarction", "has_symptom", "dyspnea"),
            ("Myocardial Infarction", "has_symptom", "sweating"),
            ("Myocardial Infarction", "has_lab_marker", "troponin"),
            ("Myocardial Infarction", "has_lab_marker", "heart_rate"),
            ("Myocardial Infarction", "has_icd10_code", "I21.9"),
            ("Myocardial Infarction", "requires_test", "12-Lead Electrocardiogram"),
            ("Myocardial Infarction", "requires_test", "Cardiac Troponin Panel"),
            ("Myocardial Infarction", "has_treatment_class", "Antiplatelets")
        ])

        # Triples mapping Acute Stroke (Neurology)
        self.triples.update([
            ("Acute Stroke", "has_symptom", "facial droop"),
            ("Acute Stroke", "has_symptom", "altered mental status"),
            ("Acute Stroke", "has_symptom", "speech difficulty"),
            ("Acute Stroke", "has_symptom", "arm weakness"),
            ("Acute Stroke", "has_icd10_code", "I63.9"),
            ("Acute Stroke", "requires_test", "Non-Contrast Head CT Scan"),
            ("Acute Stroke", "requires_test", "MRI Brain"),
            ("Acute Stroke", "has_treatment_class", "Thrombolytics"),
            
            # Patient Digital Twin Physiological Constraints
            ("Weight Loss", "reduces", "systolic_bp"),
            ("Metformin", "reduces", "blood_glucose"),
            ("Exercise", "improves", "insulin_sensitivity"),
            ("High Sodium", "increases", "systolic_bp"),
            ("Beta Blockers", "reduces", "heart_rate")
        ])

    def query_relations(self, subject: str, predicate: str = None) -> List[str]:
        """
        Queries the knowledge graph for objects connected to a subject.
        E.g. query_relations("Pneumonia", "has_symptom") -> ["cough", "fever", "dyspnea", "crackles"]
        """
        results = []
        for s, p, o in self.triples:
            if s.lower() == subject.lower():
                if predicate is None or p.lower() == predicate.lower():
                    results.append(o)
        return results

    def verify_symptoms_against_disease(self, disease: str, note_text: str) -> float:
        """
        Computes the matching coefficient representing what fraction of the disease's symptoms
        mentioned in the graph are documented in the patient notes text.
        """
        symptoms = self.query_relations(disease, "has_symptom")
        if not symptoms:
            return 0.0
            
        matched = 0
        note_lower = note_text.lower()
        for sym in symptoms:
            if sym.lower() in note_lower:
                matched += 1
                
        return float(matched / len(symptoms))

if __name__ == "__main__":
    kg = MedicalKnowledgeGraph()
    print("Medical Knowledge Graph queries:")
    print(f" - Symptoms of Pneumonia: {kg.query_relations('Pneumonia', 'has_symptom')}")
    print(f" - ICD10 of Stroke: {kg.query_relations('Acute Stroke', 'has_icd10_code')}")
    
    note = "Patient presents with high fever, intensive cough, and mild chest pain."
    match_rate = kg.verify_symptoms_against_disease("Pneumonia", note)
    print(f" - Pneumonia symptom match rate in note: {match_rate:.2%}")
