import os
from PIL import Image

from src.data.generator import PatientCraftEngine
from src.nlp.timeline import ClinicalTimelineCompiler
from src.intelligence.clinical_fuser import MultimodalIntelligenceFuser
from src.intelligence.reasoning_engine import MultimodalReasoningEngine

def run_intelligence_pipeline():
    print("="*75)
    print(" MEDTWIN AI: MULTIMODAL CLINICAL INTELLIGENCE REASONING PIPELINE ")
    print("="*75)
    
    # 1. Synthesize patient profile
    print("[1/4] Synthesizing tri-modal patient case...")
    engine = PatientCraftEngine(seed=82)
    sample = engine.generate_patient_record(has_pneumonia=True, noise_level=0.1)
    
    image = sample["image"]
    note = sample["clinical_note"]
    vitals = sample["lab_metrics"]
    
    # 2. Compile patient timeline narrative
    print("[2/4] Compiling chronological patient timelines...")
    timeline_compiler = ClinicalTimelineCompiler()
    events = [
        {"date": "2026-03-10", "type": "Note", "content": "Patient records baseline diabetes check. Blood sugar elevated."},
        {"date": "2026-07-20", "type": "Radiology Report", "content": "Chest scan reveals consolidation patches in right lung lobes."}
    ]
    timeline_narrative = timeline_compiler.generate_narrative_timeline(events)
    
    # 3. Fuse all modalities (including genomics)
    print("[3/4] Fusing notes, scans, vitals, timelines, and genomic reports...")
    fuser = MultimodalIntelligenceFuser()
    genomic_data = {
        "EGFR_mutation": "mutated",
        "BRCA1_mutation": "not_detected"
    }
    
    fused_state = fuser.fuse_patient_profile(
        image=image,
        clinical_note=note,
        vitals=vitals,
        timeline_narrative=timeline_narrative,
        genomic_data=genomic_data
    )
    
    # 4. Execute clinical reasoning
    print("[4/4] Executing clinical reasoning over fused state...")
    reasoning_engine = MultimodalReasoningEngine()
    result = reasoning_engine.evaluate_profile(fused_state)
    
    print("\n" + "="*75)
    print(" MULTIMODAL CLINICAL INTELLIGENCE DIAGNOSTIC SUMMARY ")
    print("="*75)
    print(f" PRIMARY DIAGNOSIS:  {result['diagnosis']}")
    print(f" CONFIDENCE SCORE:   {result['confidence_score']:.2%}")
    print("-"*75)
    print(" CLINICAL EVIDENCE:")
    print(f"  {result['evidence']}")
    print("-"*75)
    print(" DETECTED ACUTE RISK FACTORS:")
    for risk in result["risk_factors"]:
        print(f"  * {risk}")
    print("-"*75)
    print(" RECOMMENDED INVESTIGATIONS:")
    for inv in result["recommended_investigations"]:
        print(f"  * {inv}")
    print("-"*75)
    print(" PROPOSED TREATMENT REGIMEN (RESEARCH USE ONLY):")
    for trt in result["treatment_suggestions_research"]:
        print(f"  * {trt}")
    print("="*75)
    
    # Save text report to docs
    os.makedirs("docs", exist_ok=True)
    report_path = "docs/multimodal_intelligence_case_study.txt"
    with open(report_path, "w") as f:
        f.write("MEDTWIN AI: CLINICAL INTELLIGENCE DIAGNOSTIC CASE STUDY\n")
        f.write("="*60 + "\n")
        f.write(f"Diagnosis:    {result['diagnosis']}\n")
        f.write(f"Confidence:   {result['confidence_score']:.2%}\n")
        f.write(f"Evidence:     {result['evidence']}\n")
        f.write(f"Risk Factors: {', '.join(result['risk_factors'])}\n")
        f.write(f"Tests:        {', '.join(result['recommended_investigations'])}\n")
        f.write(f"Treatments:   {', '.join(result['treatment_suggestions_research'])}\n")
    print(f"Longitudinal case study report saved to: {report_path}")

if __name__ == "__main__":
    run_intelligence_pipeline()
