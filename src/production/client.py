import io
import json
import requests
from src.data.generator import PatientCraftEngine

def run_client_simulation(port: int = 8000):
    url = f"http://127.0.0.1:8000/predict"
    
    print("="*65)
    print("Generating simulated patient record (Infection Case)...")
    engine = PatientCraftEngine(seed=45)
    
    # Generate sick patient case (has pneumonia)
    sample = engine.generate_patient_record(has_pneumonia=True, noise_level=0.2)
    
    img = sample["image"]
    note = sample["clinical_note"]
    labs = sample["lab_metrics"]
    label = sample["label"]
    
    try:
        # Write image to in-memory PNG buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        files = {
            "file": ("radiograph.png", buffer, "image/png")
        }
        data = {
            "clinical_note": note,
            "lab_metrics_json": json.dumps(labs)
        }
        
        print(f"Submitting patient case to MedTwin AI API at {url}...")
        response = requests.post(url, files=files, data=data)
        
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
            return
            
        res_json = response.json()
        
        print("\n" + "="*65)
        print(" MEDTWIN AI CLINICAL CLIENT SIMULATION ")
        print("="*65)
        print(f"True Clinical State:     {'Pneumonia' if label == 1 else 'Normal'}")
        print(f"Predicted Diagnosis:     {res_json['diagnostic_metrics']['diagnosis_label']}")
        print(f"Base Calibrated Prob:    {res_json['diagnostic_metrics']['base_calibrated_prob']:.4f}")
        print(f"Healed Calibrated Prob:  {res_json['diagnostic_metrics']['healed_calibrated_prob']:.4f}")
        print(f"Diagnostic Set:          {res_json['diagnostic_metrics']['prediction_set']}")
        print(f"Requires Clinical Audit: {res_json['diagnostic_metrics']['requires_clinical_audit']}")
        print("-"*65)
        print("Clinical Self-Healing Reasoning Trace:")
        print(f" - Action taken:         {res_json['self_healing_trace']['action_taken']}")
        print(f" - Log:                  {res_json['self_healing_trace']['reasoning_log']}")
        print(f" - Vitals Audit:         {res_json['self_healing_trace']['physiological_audit']['details']}")
        print(f" - Lesion Density Score: {res_json['self_healing_trace']['visual_lesion_audit']['mean_pathology_score']:.4f}")
        print("="*65)
        
        # --- Consensus API Call ---
        consensus_url = f"http://127.0.0.1:8000/consensus"
        buffer.seek(0)
        files_c = {
            "file": ("radiograph.png", buffer, "image/png")
        }
        print(f"\nSubmitting patient case to MedTwin AI Consensus API at {consensus_url}...")
        response_c = requests.post(consensus_url, files=files_c, data=data)
        if response_c.status_code == 200:
            res_c_json = response_c.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI CLINICAL CONSENSUS BOARD RESULTS ")
            print("="*65)
            c_metrics = res_c_json["consensus_metrics"]
            print(f" Consensus Diagnosis:   {c_metrics['consensus_diagnosis']} ({c_metrics['icd10_code']})")
            print(f" Agreement Rate:       {c_metrics['agreement_rate']:.2%}")
            print(f" Uncertainty:          {c_metrics['consensus_uncertainty']:.2%}")
            print(f" Clinical Audit Req:   {c_metrics['requires_clinical_audit']}")
            print("-"*65)
            print("Guideline Citation from Medical RAG:")
            print(f" - Citation:           {res_c_json['guideline_citations']['citation']}")
            print(f" - Guideline:          \"{res_c_json['guideline_citations']['text'][:90]}...\"")
            print("="*65)
        else:
            print(f"Error: Consensus API returned status code {response_c.status_code}")
            print(response_c.text)
            
        # --- Patient Digital Twin Simulation API Call ---
        twin_url = f"http://127.0.0.1:8000/twin/simulate"
        # Mock obese diabetic vitals
        vitals_payload = {
            "weight": 95.0,
            "blood_pressure": 142.0,
            "blood_glucose": 170.0,
            "heart_rate": 82.0,
            "wbc_count": 7800.0,
            "oxygen_saturation": 96.0
        }
        scenario_payload = {
            "weight_change": -10.0,
            "metformin_mg": 1000.0,
            "beta_blocker_mg": 0.0,
            "exercise_hours": 4.0,
            "sodium_intake_g": 1.5
        }
        twin_data = {
            "patient_vitals_json": json.dumps(vitals_payload),
            "what_if_scenario_json": json.dumps(scenario_payload)
        }
        
        print(f"\nSubmitting digital twin query to {twin_url}...")
        response_t = requests.post(twin_url, data=twin_data)
        if response_t.status_code == 200:
            res_t_json = response_t.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI PATIENT DIGITAL TWIN SIMULATION ")
            print("="*65)
            print(f" Cardiovascular Risk: {res_t_json['risk_scores']['cardiovascular_risk']:.2%}")
            print(f" Readmission Risk:    {res_t_json['risk_scores']['hospital_readmission_risk']:.2%}")
            print(f" Projected Recovery:   {res_t_json['risk_scores']['predicted_recovery_time_days']:.1f} days")
            print(f" Complications:       {res_t_json['complications']}")
            print("-"*65)
            print("RAG Citations:")
            for cit in res_t_json["guideline_citations"]:
                print(f" * {cit}")
            print("="*65)
        else:
            print(f"Error: Digital Twin API returned status code {response_t.status_code}")
            print(response_t.text)
            
        # --- VLM Inference API Call (Florence-2 test) ---
        vlm_url = f"http://127.0.0.1:8000/vlm/infer"
        buffer.seek(0)
        files_v = {
            "file": ("radiograph.png", buffer, "image/png")
        }
        vlm_data = {
            "model_name": "florence-2",
            "prompt": "ground lung consolidation opacity",
            "task": "grounding"
        }
        print(f"\nSubmitting VLM query to {vlm_url}...")
        response_v = requests.post(vlm_url, files=files_v, data=vlm_data)
        if response_v.status_code == 200:
            res_v_json = response_v.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI CLINICAL VLM RUNNER ")
            print("="*65)
            print(f" Model Used:      {res_v_json['model']}")
            print(f" Task Performed:  {res_v_json['task']}")
            print(f" Grounded Boxes:  {res_v_json['outputs']['grounded_box']}")
            print(f" Visual Caption:  {res_v_json['outputs']['caption']}")
            print("="*65)
        else:
            print(f"Error: VLM API returned status code {response_v.status_code}")
            print(response_v.text)
            
        # --- Fine-Tuning Job Trigger API Call ---
        ft_run_url = f"http://127.0.0.1:8000/finetune/run"
        ft_status_url = f"http://127.0.0.1:8000/finetune/status"
        
        print(f"\nTriggering asynchronous fine-tuning job at {ft_run_url}...")
        response_ft = requests.post(ft_run_url)
        if response_ft.status_code == 200:
            print("Fine-tuning job successfully submitted. Checking status...")
            response_st = requests.get(ft_status_url)
            st_json = response_st.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI ADAPTER TRAINING STATUS ")
            print("="*65)
            print(f" Job Status:        {st_json['status']}")
            print(f" Current Epoch:     {st_json['epoch']}")
            print(f" Training Loss:     {st_json['loss']:.4f}")
            print(f" Val Accuracy:      {st_json['accuracy']:.2%}")
            print("="*65)
        else:
            print(f"Error: Fine-tuning API returned status code {response_ft.status_code}")
            print(response_ft.text)
            
        # --- NLP RAG Query API Call ---
        nlp_query_url = f"http://127.0.0.1:8000/nlp/query"
        nlp_payload = {
            "query": "What is the recommended treatment for Pneumonia according to guidelines?"
        }
        print(f"\nSubmitting RAG query to {nlp_query_url}...")
        response_nlp = requests.post(nlp_query_url, data=nlp_payload)
        if response_nlp.status_code == 200:
            res_nlp_json = response_nlp.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI CLINICAL NLP RAG RESPONSE ")
            print("="*65)
            print(f" Query submitted:  {res_nlp_json['query']}")
            print(f" Generated Answer: {res_nlp_json['answer']}")
            print(f" RAG Faithfulness: {res_nlp_json['evaluation_metrics']['faithfulness_score']:.2%}")
            print(f" Hallucination:    {res_nlp_json['evaluation_metrics']['hallucination_rate']:.2%}")
            if len(res_nlp_json["drug_interaction_warnings"]) > 0:
                print(f" Warnings:         {res_nlp_json['drug_interaction_warnings']}")
            print("="*65)
        else:
            print(f"Error: NLP RAG API returned status code {response_nlp.status_code}")
            print(response_nlp.text)
            
        # --- Multimodal Clinical Intelligence API Call ---
        intel_url = "http://127.0.0.1:8000/intelligence/diagnose"
        intel_payload = {
            "clinical_note": "Patient presents with cough and fever. Scheduled for abdominal CT scan with Contrast Dye.",
            "lab_metrics_json": json.dumps({"age": 62, "temperature": 101.2, "heart_rate": 92, "wbc_count": 14500, "oxygen_saturation": 94, "blood_pressure": 142, "blood_glucose": 165}),
            "timeline_events_json": json.dumps([
                {"date": "2026-03-10", "type": "Note", "content": "Diabetes tracking shows hyperglycemia."}
            ]),
            "genomic_data_json": json.dumps({"EGFR_mutation": "mutated"})
        }
        print(f"\nSubmitting Multimodal case file to {intel_url}...")
        response_intel = requests.post(intel_url, data=intel_payload)
        if response_intel.status_code == 200:
            res_intel_json = response_intel.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI MULTIMODAL CLINICAL INTELLIGENCE DIAGNOSIS ")
            print("="*65)
            print(f" DIAGNOSIS:  {res_intel_json['diagnosis']}")
            print(f" CONFIDENCE:  {res_intel_json['confidence_score']:.2%}")
            print(f" EVIDENCE:    {res_intel_json['evidence']}")
            print(f" RISKS:       {res_intel_json['risk_factors']}")
            print(f" TESTS:       {res_intel_json['recommended_investigations']}")
            print(f" TREATMENTS:  {res_intel_json['treatment_suggestions_research']}")
            print("="*65)
        else:
            print(f"Error: Intelligence API returned status code {response_intel.status_code}")
            print(response_intel.text)
            
        # --- Multi-Agent Clinical Consensus API Call ---
        consensus_url = "http://127.0.0.1:8000/agents/consensus"
        consensus_payload = {
            "clinical_note": "Patient presents with cough and fever. Scheduled for abdominal CT scan with Contrast Dye.",
            "lab_metrics_json": json.dumps({"age": 62, "temperature": 101.2, "heart_rate": 92, "wbc_count": 14500, "oxygen_saturation": 94, "blood_pressure": 142, "blood_glucose": 165})
        }
        print(f"\nSubmitting case file to agent consensus board at {consensus_url}...")
        response_cons = requests.post(consensus_url, data=consensus_payload)
        if response_cons.status_code == 200:
            res_cons_json = response_cons.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI MULTI-AGENT BOARD CONSENSUS ")
            print("="*65)
            print(f" CONSENSUS:   {res_cons_json['consensus_diagnosis']} ({res_cons_json['icd10_code']})")
            print(f" AGREEMENT:   {res_cons_json['agreement_rate']:.2%}")
            print(f" UNCERTAINTY: {res_cons_json['consensus_uncertainty']:.2%}")
            print(f" RAG CITATION: {res_cons_json['literature_citation']}")
            print(f" DISAGREEMENTS: {len(res_cons_json['disagreements'])} agents")
            print("="*65)
        else:
            print(f"Error: Consensus API returned status code {response_cons.status_code}")
            print(response_cons.text)
            
        # --- Medical Imaging API Call ---
        from PIL import Image
        imaging_url = "http://127.0.0.1:8000/imaging/analyze"
        # Create a mock image file
        mock_img_io = io.BytesIO()
        Image.new("RGB", (256, 256), color="grey").save(mock_img_io, format="PNG")
        mock_img_io.seek(0)
        
        files_payload = {
            "file": ("mock_scan.png", mock_img_io, "image/png")
        }
        data_payload = {
            "modality": "mri",
            "task": "classify"
        }
        print(f"\nSubmitting brain MRI scan to {imaging_url}...")
        response_img = requests.post(imaging_url, data=data_payload, files=files_payload)
        if response_img.status_code == 200:
            res_img_json = response_img.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI IMAGING CLASSIFICATION ")
            print("="*65)
            print(f" Modality Routed: {res_img_json['modality']}")
            print(f" Task Type:      {res_img_json['task']}")
            print(f" Diagnosis:      {res_img_json['diagnosis']}")
            print(f" Confidence:     {res_img_json['confidence']:.2%}")
            print("="*65)
        else:
            print(f"Error: Imaging API returned status code {response_img.status_code}")
            print(response_img.text)
            
        # --- Medical OCR API Call ---
        ocr_url = "http://127.0.0.1:8000/ocr/analyze"
        mock_doc_io = io.BytesIO()
        Image.new("RGB", (300, 100), color="white").save(mock_doc_io, format="PNG")
        mock_doc_io.seek(0)
        
        files_payload_ocr = {
            "file": ("mock_prescription.png", mock_doc_io, "image/png")
        }
        data_payload_ocr = {
            "document_type": "prescription"
        }
        print(f"\nSubmitting scanned prescription to {ocr_url}...")
        response_ocr = requests.post(ocr_url, data=data_payload_ocr, files=files_payload_ocr)
        if response_ocr.status_code == 200:
            res_ocr_json = response_ocr.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI DOCUMENT OCR ")
            print("="*65)
            print(f" Document type:    {res_ocr_json['document_type']}")
            print(f" OCR Text parsed:  \"{res_ocr_json['ocr_text']}\"")
            print(f" Extracted Meds:   {res_ocr_json['extracted_medications']}")
            print(f" Extracted Doses:  {res_ocr_json['extracted_dosages']}")
            print("="*65)
        else:
            print(f"Error: OCR API returned status code {response_ocr.status_code}")
            print(response_ocr.text)
            
        # --- Synthetic Generation API Call ---
        synth_url = "http://127.0.0.1:8000/synthetic/generate"
        synth_payload = {
            "modality": "mri",
            "condition": "alkaptonuria",
            "quality": "blurred"
        }
        print(f"\nRequesting synthetic data generation for Alkaptonuria at {synth_url}...")
        response_synth = requests.post(synth_url, data=synth_payload)
        if response_synth.status_code == 200:
            res_synth_json = response_synth.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI SYNTHETIC DATA GENERATION ")
            print("="*65)
            print(f" Condition:       {res_synth_json['condition']}")
            print(f" Modality:        {res_synth_json['modality']}")
            print(f" Prescription:    \"{res_synth_json['synthetic_prescription']}\"")
            print(f" Lab Report:      \"{res_synth_json['synthetic_lab_report']}\"")
            print(f" Base64 Image:    {res_synth_json['synthetic_image_b64'][:40]}... (Truncated)")
            print("="*65)
        else:
            print(f"Error: Synthetic API returned status code {response_synth.status_code}")
            print(response_synth.text)
            
        # --- Explainability & Failure Analysis API Call ---
        explain_url = "http://127.0.0.1:8000/explainability/analyze"
        explain_payload = {
            "prediction": "Normal",
            "ground_truth": "Pneumonia",
            "ocr_text": "Rx: Ceftriaxone 1g IV daily.",
            "rag_similarity": 0.85,
            "ner_faithfulness": 0.95,
            "vitals_json": json.dumps({"temperature": 101.5, "wbc_count": 14500, "oxygen_saturation": 92.0}),
            "clinical_note": "Patient presents with cough and fever."
        }
        print(f"\nSubmitting diagnostic failure and query parameters to {explain_url}...")
        response_exp = requests.post(explain_url, data=explain_payload)
        if response_exp.status_code == 200:
            res_exp_json = response_exp.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI EXPLAINABILITY & FAILURE ANALYSIS ")
            print("="*65)
            print(f" Failure Detected: {res_exp_json['failure_audit']['failure_detected']}")
            print(f" Failure Class:    {res_exp_json['failure_audit']['failure_class']}")
            print(f" Root Cause:       \"{res_exp_json['failure_audit']['root_cause_analysis']}\"")
            print(f" Retraining Plan:  \"{res_exp_json['failure_audit']['suggested_retraining_action']}\"")
            print(f" SHAP Attributions: {res_exp_json['shap_values']}")
            print(f" Grad-CAM Layer:   {res_exp_json['grad_cam']['target_layer']}")
            print("="*65)
        else:
            print(f"Error: Explainability API returned status code {response_exp.status_code}")
            print(response_exp.text)
            
        # --- Experiment Tracking API Call ---
        track_url = "http://127.0.0.1:8000/tracking/log"
        track_payload = {
            "run_name": "Prod_LoRA_Rank_16",
            "params_json": json.dumps({"lora_rank": 16, "learning_rate": 3e-4, "model_version": "v1.2"}),
            "metrics_json": json.dumps({"peak_vram_gb": 16.2, "validation_f1": 0.945, "latency_ms": 45.0})
        }
        print(f"\nSubmitting experiment run metrics to tracker at {track_url}...")
        response_tr = requests.post(track_url, data=track_payload)
        if response_tr.status_code == 200:
            res_tr_json = response_tr.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI EXPERIMENT RUN TRACKING ")
            print("="*65)
            print(f" Status:           {res_tr_json['status']}")
            print(f" Logged Run:       {res_tr_json['logged_run']}")
            print(f" Parameters:       {res_tr_json['tracked_parameters']}")
            print(f" Metrics:          {res_tr_json['tracked_metrics']}")
            print("="*65)
        else:
            print(f"Error: Tracking API returned status code {response_tr.status_code}")
            print(response_tr.text)
            
        # --- Distributed Training & Optimization API Call ---
        opt_url = "http://127.0.0.1:8000/optimization/benchmark"
        print(f"\nRequesting runtime optimization benchmarks from {opt_url}...")
        response_opt = requests.post(opt_url)
        if response_opt.status_code == 200:
            res_opt_json = response_opt.json()
            print("\n" + "="*65)
            print(" MEDTWIN AI RUNTIME OPTIMIZATION BENCHMARK ")
            print("="*65)
            print(f" Status:           {res_opt_json['status']}")
            print(f" Saved Plot Path:  {res_opt_json['benchmark_plot_saved']}")
            print(f" TensorRT Latency: {res_opt_json['inference_benchmarks']['TensorRT']['latency_ms']} ms")
            print(f" TensorRT Throughput: {res_opt_json['inference_benchmarks']['TensorRT']['throughput']} inf/sec")
            print("="*65)
        else:
            print(f"Error: Optimization API returned status code {response_opt.status_code}")
            print(response_opt.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Connection Refused. Please start the FastAPI server first using:")
        print("  python -m uvicorn src.production.app:app --host 127.0.0.1 --port 8000")

if __name__ == "__main__":
    run_client_simulation()
