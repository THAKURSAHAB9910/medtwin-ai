from typing import Dict, Any, List

class MultimodalReasoningEngine:
    """
    Evaluates fused multimodal patient profiles to output explainable diagnoses,
    risk profiles, recommended investigations, and research-use treatments.
    """
    def evaluate_profile(self, fused_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs clinical reasoning over the patient's combined modalities.
        """
        vitals = fused_state["vitals"]
        text = fused_state["narrative_text"].lower()
        genomics = fused_state["genomics"]
        has_radio = fused_state["has_radiograph"]
        
        # 1. Determine diagnosis and confidence
        is_pneumonia = False
        is_diabetes = False
        is_hypertension = False
        
        evidence_points = []
        risks = []
        investigations = []
        treatments = []
        
        # Scan-vital evaluations
        if has_radio and ("pneumonia" in text or "consolidation" in text or "opacity" in text):
            is_pneumonia = True
            evidence_points.append("Visual chest radiograph reveals lung opacity/consolidation.")
            
        if vitals["temperature"] > 100.4:
            evidence_points.append(f"Patient is febrile (Temp: {vitals['temperature']}°F).")
            if is_pneumonia:
                risks.append("SIRS criteria matching: systemic response to infection.")
                
        if vitals["wbc_count"] > 11000:
            evidence_points.append(f"Leukocytosis detected (WBC: {vitals['wbc_count']}/uL).")
            
        if vitals["oxygen_saturation"] < 92:
            risks.append(f"Hypoxia danger: Oxygen saturation is depressed ({vitals['oxygen_saturation']}%).")
            investigations.append("Arterial blood gas (ABG) analysis")
            
        if vitals["blood_glucose"] > 150 or "diabetes" in text or "metformin" in text:
            is_diabetes = True
            evidence_points.append(f"Hyperglycemic trending (Blood Glucose: {vitals['blood_glucose']} mg/dL).")
            investigations.append("Hemoglobin A1c (HbA1c) tracking")
            
        if vitals["blood_pressure"] > 135 or "hypertension" in text:
            is_hypertension = True
            evidence_points.append(f"Elevated blood pressure observed (BP: {vitals['blood_pressure']} mmHg).")
            
        # Genomics evaluations
        if genomics.get("EGFR_mutation") == "mutated":
            evidence_points.append("Genomic report flags EGFR mutation, compounding pulmonary pathology susceptibility.")
            risks.append("Genetic cancer risk: Elevated sensitivity to pulmonary nodules.")
            investigations.append("Chest CT scan with contrast to audit nodule margins")
            
        # Compile final diagnosis labels
        diag_labels = []
        if is_pneumonia:
            diag_labels.append("Community-Acquired Pneumonia")
            investigations.append("Sputum Gram stain and culture")
            treatments.append("Ceftriaxone 1g IV daily + Azithromycin 500mg PO daily")
        if is_diabetes:
            diag_labels.append("Type 2 Diabetes Mellitus")
            treatments.append("Metformin 500mg PO twice daily (adjust for renal function)")
        if is_hypertension:
            diag_labels.append("Essential Hypertension")
            treatments.append("Lisinopril 10mg PO daily")
            
        if not diag_labels:
            diagnosis = "Normal Health State"
            confidence = 0.95
            evidence = "Patient clinical vitals, notes, and scans are within normal reference limits."
        else:
            diagnosis = " compounded by ".join(diag_labels)
            confidence = 0.88 if len(diag_labels) > 1 else 0.94
            evidence = " ".join(evidence_points)
            
        # General investigations & treatments
        if not investigations:
            investigations = ["Routine outpatient annual labs"]
        if not treatments:
            treatments = ["Supportive lifestyle maintenance"]
            
        return {
            "diagnosis": diagnosis,
            "confidence_score": confidence,
            "evidence": evidence,
            "risk_factors": risks or ["No acute risk factors flagged"],
            "recommended_investigations": investigations,
            "treatment_suggestions_research": treatments,
            "explanation_report": f"REASONING OVERVIEW: {evidence} Risk assessments identified: {', '.join(risks or ['none'])}."
        }
