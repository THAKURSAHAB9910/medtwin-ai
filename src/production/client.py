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
            
    except requests.exceptions.ConnectionError:
        print("\nError: Connection Refused. Please start the FastAPI server first using:")
        print("  python -m uvicorn src.production.app:app --host 127.0.0.1 --port 8000")

if __name__ == "__main__":
    run_client_simulation()
