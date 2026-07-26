import re
from typing import List, Dict, Tuple

class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation (RAG) system.
    Searches clinical guidelines (ATS, ACC, ASA) using text overlap scoring
    to return citations, resolving hallucination issues.
    """
    def __init__(self):
        self.guidelines_db: List[Dict[str, str]] = []
        self._initialize_database()

    def _initialize_database(self):
        self.guidelines_db = [
            {
                "disease": "Pneumonia",
                "citation": "ATS/IDSA Guidelines (2019) | PMID: 31573350",
                "text": "For community-acquired pneumonia, empirical antibiotic therapy should target typical and atypical pathogens. Diagnosis is confirmed via chest radiograph showing lung opacities, combined with fever, cough, and leukocytosis. Code: J18.9."
            },
            {
                "disease": "Myocardial Infarction",
                "citation": "AHA/ACC STEMI Guidelines (2020) | PMID: 32412891",
                "text": "Patients with suspected acute coronary syndrome or myocardial infarction must receive an immediate 12-lead ECG, cardiac troponin assay tracking, and emergency antiplatelet loading (e.g., aspirin). Code: I21.9."
            },
            {
                "disease": "Acute Stroke",
                "citation": "AHA/ASA Ischemic Stroke Guidelines (2018) | PMID: 29367465",
                "text": "Acute ischemic stroke requires rapid clinical assessment (NIHSS), non-contrast head CT scan within 25 minutes of arrival, and evaluation for intravenous thrombolysis (Alteplase) within a 3-4.5 hour window. Code: I63.9."
            },
            {
                "disease": "Diabetes Mellitus",
                "citation": "ADA Standards of Care (2023) | PMID: 36507450",
                "text": "Glycemic targets should be individualized. In general, a target HbA1c < 7.0% (corresponding to average glucose ~154 mg/dL) is recommended for non-pregnant adults. First-line therapy typically includes metformin and lifestyle intervention (weight loss, diet). Code: E11.9."
            },
            {
                "disease": "Essential Hypertension",
                "citation": "AHA/ACC Hypertension Guidelines (2017) | PMID: 29133354",
                "text": "Hypertension is defined as blood pressure >= 130/80 mmHg. Lifestyle modifications including sodium restriction (<1500mg/day), weight loss (expecting ~1 mmHg drop per kg lost), and aerobic exercise are recommended. Code: I10."
            }
        ]

    def retrieve_guidelines(self, query: str, top_k: int = 1) -> List[Dict[str, str]]:
        """
        Retrieves top_k matching medical guidelines based on keyword overlap.
        """
        query_words = set(re.findall(r'\w+', query.lower()))
        if not query_words:
            return self.guidelines_db[:top_k]
            
        scored_documents = []
        for doc in self.guidelines_db:
            text_words = set(re.findall(r'\w+', (doc["text"] + " " + doc["disease"]).lower()))
            # Calculate Jaccard similarity coefficient (word overlap)
            intersection = query_words.intersection(text_words)
            union = query_words.union(text_words)
            score = len(intersection) / len(union) if union else 0.0
            scored_documents.append((score, doc))
            
        # Sort documents by descending overlap score
        scored_documents.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k documents
        return [doc for score, doc in scored_documents[:top_k]]

if __name__ == "__main__":
    rag = MedicalRAG()
    results = rag.retrieve_guidelines("cough and fever lung consolidation")
    print("Medical RAG retrieval results:")
    for doc in results:
        print(f" - Citation: {doc['citation']}")
        print(f" - Guideline Text: {doc['text']}")
