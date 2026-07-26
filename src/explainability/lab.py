from typing import Dict, Any, List

class MedicalFailureAnalysisLab:
    """
    Diagnostic auditor that isolates and classifies clinical model failures,
    outputting root causes, retraining suggestions, and data augmentation plans.
    """
    def audit_prediction(self, case_record: Dict[str, Any], evaluation_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspects output predictions against ground truth logs to isolate failure nodes.
        """
        pred = case_record.get("prediction", "").lower().strip()
        truth = case_record.get("ground_truth", "").lower().strip()
        rag_sim = evaluation_metrics.get("rag_similarity", 1.0)
        ner_faithfulness = evaluation_metrics.get("ner_faithfulness", 1.0)
        ocr_text = case_record.get("ocr_text", "valid")
        
        # 1. Detect Failure Class
        failure_class = "None"
        root_cause = "Model executed within safe margins. Prediction matches ground truth."
        retrain = "No retraining needed."
        augment = "No augmentation needed."
        
        if pred != truth:
            if pred == "normal" and truth != "normal":
                failure_class = "False Negative"
                root_cause = "Model missed active pathologic consolidation indicator signs in clinical scan."
                retrain = "Retrain model with focal loss weighted heavily on positive pathology examples."
                augment = "Augment dataset with localized bbox overlays of positive classes."
            elif pred != "normal" and truth == "normal":
                failure_class = "False Positive"
                root_cause = "Model flagged healthy variant tissue as an active anomaly."
                retrain = "Expose model to wider negative cohort variants during train iterations."
                augment = "Augment negative datasets with varying illumination and contrast ranges."
            else:
                failure_class = "Misdiagnosis"
                root_cause = f"Model classified condition as {pred} instead of the true {truth} label."
                retrain = "Increase cross-entropy weights for low-frequency classes."
                augment = "Synthesize rare disease samples matching the missed condition class."
                
        elif not ocr_text or "error" in ocr_text.lower():
            failure_class = "OCR Error"
            root_cause = "Document scanner outputted blank or severely misspelled text tokens."
            retrain = "Re-train TrOCR decoder using clinical handwriting datasets."
            augment = "Augment document scans with random elastic transforms and crump noise."
            
        elif ner_faithfulness < 0.70:
            failure_class = "Hallucination"
            root_cause = f"Model generated treatment output containing unsupported clinical terms (Faithfulness: {ner_faithfulness:.2%})."
            retrain = "Finetune generator weights on strict guideline QA context templates."
            augment = "Augment prompts with local vector DB embedding context blocks."
            
        elif rag_sim < 0.30:
            failure_class = "Retrieval Failure"
            root_cause = f"Vector database failed to retrieve relevant guideline context (Similarity: {rag_sim:.2f})."
            retrain = "Optimize vector database index parameters (increase top-k or use reranker)."
            augment = "Finetune MedicalEmbeddingGenerator using contrastive medical triplet losses."
            
        elif "missing_evidence" in case_record and case_record["missing_evidence"]:
            failure_class = "Missing Evidence"
            root_cause = "Model outputted high confidence diagnostic labels without citing supporting symptom indicators."
            retrain = "Integrate multi-task training forcing co-prediction of diagnosis and justification strings."
            augment = "Synthesize training pairs detailing explicit symptom-to-diagnosis evidence traces."
            
        return {
            "failure_detected": failure_class != "None",
            "failure_class": failure_class,
            "root_cause_analysis": root_cause,
            "suggested_retraining_action": retrain,
            "suggested_augmentation_strategy": augment
        }
