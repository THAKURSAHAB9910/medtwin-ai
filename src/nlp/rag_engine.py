from typing import Dict, Any, List

from src.nlp.vector_db import LocalVectorDB
from src.nlp.entity_extractor import ClinicalNER

class ClinicalRAGEngine:
    """
    Coordinates local vector databases and Named Entity Recognition to answer
    clinical queries, check drug contraindications, and audit hallucination rates.
    """
    def __init__(self, vector_db: LocalVectorDB, ner: ClinicalNER):
        self.vector_db = vector_db
        self.ner = ner

    def query_clinical_rag(self, query: str) -> Dict[str, Any]:
        """
        Retrieves context documents, answers the query, and evaluates faithfulness metrics.
        """
        # 1. Semantic Retrieval
        retrieved_docs = self.vector_db.search(query, k=2)
        
        # Aggregate retrieved source text
        context_text = " ".join([d["text"] for d in retrieved_docs])
        
        # 2. Entity Extractions
        query_entities = self.ner.extract_entities(query)
        context_entities = self.ner.extract_entities(context_text)
        
        # 3. Generate Clinical Answer (Rule-based clinical synthesis of retrieved nodes)
        # Check for matching medical guidelines in context
        answer = "No matching clinical guidelines retrieved."
        hallucination_offset = 0.0
        
        if "pneumonia" in query.lower():
            if "ceftriaxone" in context_text.lower():
                answer = "Based on retrieved ADA/AHA clinical protocols, Ceftriaxone 1g IV daily is the recommended primary therapy for community-acquired pneumonia."
            else:
                # If we recommend Ceftriaxone but it is NOT in the retrieved context, this is a hallucination!
                answer = "We recommend Ceftriaxone 1g IV daily for pneumonia."
                hallucination_offset = 0.40 # Flag that we generated fact outside context!
        elif "diabetes" in query.lower() or "metformin" in query.lower():
            if "metformin" in context_text.lower():
                answer = "Standard guidelines specify Metformin 500mg PO twice daily as baseline therapy for diabetes mellitus."
            else:
                answer = "Standard guidelines specify Metformin 1000mg PO daily."
                hallucination_offset = 0.35
                
        # 4. Check for drug interactions in query and answer
        all_meds = list(set(query_entities["medications"] + context_entities["medications"]))
        interactions = self.ner.check_drug_interactions(all_meds)
        
        # 5. Evaluate Faithfulness and Hallucination Rates
        answer_entities = self.ner.extract_entities(answer)
        
        # Faithfulness = fraction of answer entities supported in retrieved context
        supported_entities = 0
        total_answer_entities = len(answer_entities["medications"]) + len(answer_entities["diseases"])
        
        if total_answer_entities > 0:
            for med in answer_entities["medications"]:
                if med.lower() in context_text.lower():
                    supported_entities += 1
            for dis in answer_entities["diseases"]:
                if dis.lower() in context_text.lower():
                    supported_entities += 1
            faithfulness = float(supported_entities / total_answer_entities)
        else:
            faithfulness = 1.0
            
        # Adjust faithfulness based on known hallucination offsets
        faithfulness = max(0.0, faithfulness - hallucination_offset)
        hallucination_rate = 1.0 - faithfulness
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_context": [d["text"] for d in retrieved_docs],
            "drug_interaction_warnings": interactions,
            "evaluation_metrics": {
                "retrieved_documents_count": len(retrieved_docs),
                "faithfulness_score": faithfulness,
                "hallucination_rate": hallucination_rate,
                "hallucination_flag": hallucination_rate > 0.15
            }
        }
        
    def evaluate_pipeline_quality(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Runs quality audits over test queries, compiling average hallucination rates.
        """
        faithfulness_scores = []
        hallucination_rates = []
        
        for q in test_queries:
            res = self.query_clinical_rag(q)
            metrics = res["evaluation_metrics"]
            faithfulness_scores.append(metrics["faithfulness_score"])
            hallucination_rates.append(metrics["hallucination_rate"])
            
        return {
            "average_faithfulness": float(np.mean(faithfulness_scores)),
            "average_hallucination_rate": float(np.mean(hallucination_rates))
        }

import numpy as np
