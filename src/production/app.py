import io
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response, HTMLResponse
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from prometheus_client import Counter, Summary, generate_latest, CONTENT_TYPE_LATEST

from src.training.trainer import MedTwinAIModule
from src.training.self_healing import ClinicalSelfHealingEngine
from src.agents.coordinator import CoordinatorAgent
from src.twin.simulation import PatientDigitalTwinEngine

app = FastAPI(
    title="MedTwin AI: Production Multimodal Healthcare Foundation API",
    description="Research-Grade Multimodal Diagnostic Decision Support System",
    version="1.0.0"
)

# Prometheus counters
REQUEST_COUNT = Counter("medtwin_requests_total", "Total requests received", ["endpoint"])
LATENCY_SUMMARY = Summary("medtwin_latency_seconds", "Inference latency in seconds")
HEALED_COUNT = Counter("medtwin_self_healing_total", "Total self-healing operations triggered", ["action"])

model: Optional[MedTwinAIModule] = None
self_healing_engine: Optional[ClinicalSelfHealingEngine] = None
coordinator_agent: Optional[CoordinatorAgent] = None
twin_engine: Optional[PatientDigitalTwinEngine] = None
img_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

@app.on_event("startup")
def load_model():
    global model, self_healing_engine, coordinator_agent, twin_engine
    print("Initializing MedTwin AI foundation models...")
    model = MedTwinAIModule(use_mock_vlm=True)
    model.eval()
    
    # Pre-fit uncertainty calibrator for production stability
    mock_logits = torch.randn(20)
    mock_labels = torch.randint(0, 2, (20,))
    model.calibrator.fit_temperature(mock_logits, mock_labels, epochs=5)
    model.calibrator.fit_conformal(
        model.calibrator.get_calibrated_probability(mock_logits).detach().numpy(),
        mock_labels.numpy(),
        alpha=0.1
    )
    
    self_healing_engine = ClinicalSelfHealingEngine()
    coordinator_agent = CoordinatorAgent()
    twin_engine = PatientDigitalTwinEngine()
    print("Models, calibrator, multi-agent coordinator, and digital twin engine initialized.")

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    try:
        with open("src/production/static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")

@app.get("/health")
def health_check():
    if model is not None and self_healing_engine is not None:
        return {"status": "healthy", "clinical_modules_ready": True}
    raise HTTPException(status_code=503, detail="Clinical Model modules not initialized")

@app.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict_case(
    file: UploadFile = File(...),
    clinical_note: str = Form(...),
    lab_metrics_json: str = Form(...) # Expects {"age": X, "temperature": X, "heart_rate": X, "wbc_count": X, "oxygen_saturation": X}
):
    REQUEST_COUNT.labels(endpoint="/predict").inc()
    start_time = time.perf_counter()
    
    if model is None or self_healing_engine is None:
        raise HTTPException(status_code=503, detail="Clinical server offline")
        
    try:
        # 1. Image loading
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_tensor = img_transform(pil_img).unsqueeze(0).to(model.device)
        
        # 2. Lab metrics processing
        labs_dict = json.loads(lab_metrics_json)
        labs_vector = torch.tensor([[
            labs_dict["age"],
            labs_dict["temperature"],
            labs_dict["heart_rate"],
            labs_dict["wbc_count"],
            labs_dict["oxygen_saturation"]
        ]], dtype=torch.float32).to(model.device)
        
        # 3. Model inference forward
        with torch.no_grad():
            outputs = model(img_tensor, [clinical_note], labs_vector)
            
            raw_logit = outputs["fused_logit"]
            prob = float(model.calibrator.get_calibrated_probability(raw_logit).squeeze().cpu().numpy())
            lesion_map = torch.sigmoid(outputs["lesion_map"]).squeeze().cpu().numpy()
            
        # 4. Clinical self-healing loop
        healed_res = self_healing_engine.verify_and_heal(prob, lesion_map, labs_dict)
        HEALED_COUNT.labels(action=healed_res["action_taken"]).inc()
        
        # 5. Conformal Prediction diagnostic set
        conformal_set = model.calibrator.predict_conformal_set(np.array([healed_res["healed_prob"]]))[0]
        
        result = {
            "diagnostic_metrics": {
                "base_calibrated_prob": prob,
                "healed_calibrated_prob": healed_res["healed_prob"],
                "prediction_set": conformal_set, # 0 = Normal, 1 = Pneumonia
                "diagnosis_label": "Pneumonia" if healed_res["healed_label"] == 1 else "Normal",
                "is_active_disease": bool(healed_res["healed_label"]),
                "requires_clinical_audit": len(conformal_set) > 1
            },
            "self_healing_trace": {
                "action_taken": healed_res["action_taken"],
                "reasoning_log": healed_res["reason"],
                "physiological_audit": healed_res["physiological_check"],
                "visual_lesion_audit": {
                    "mean_pathology_score": healed_res["visual_check"]["mean_pathology_score"],
                    "has_visual_lesion": healed_res["visual_check"]["has_visual_lesion"]
                }
            }
        }
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return result
        
    except KeyError as ke:
        raise HTTPException(status_code=400, detail=f"Missing vital marker in JSON keys: {str(ke)}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON vital metrics format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clinical inference error: {str(e)}")

@app.post("/consensus")
async def predict_consensus(
    clinical_note: str = Form(...),
    lab_metrics_json: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    """
    Accepts clinical inputs and runs multi-agent clinical consensus reasoning.
    """
    REQUEST_COUNT.labels(endpoint="/consensus").inc()
    start_time = time.perf_counter()
    
    if coordinator_agent is None:
        raise HTTPException(status_code=503, detail="Consensus engine offline")
        
    try:
        # Load vitals
        labs_dict = json.loads(lab_metrics_json)
        
        patient_data = {
            "clinical_note": clinical_note,
            "lab_metrics": labs_dict
        }
        if file is not None:
            patient_data["image"] = True
            
        # Execute Multi-Agent State Flow
        state = coordinator_agent.run_consensus(patient_data)
        res = state.coordinator_consensus
        
        # Format opinions as dictionaries
        opinions_serialized = {}
        for role, opinion in state.specialist_opinions.items():
            opinions_serialized[role] = opinion.dict()
            
        result = {
            "consensus_metrics": {
                "consensus_diagnosis": res["consensus_diagnosis"],
                "icd10_code": res["icd10_code"],
                "consensus_uncertainty": res["consensus_uncertainty"],
                "agreement_rate": res["agreement_rate"],
                "requires_clinical_audit": res["consensus_uncertainty"] > 0.35
            },
            "consensus_details": {
                "evidence_ranking": res["evidence_ranking"],
                "disagreements": res["disagreements"],
                "final_tests": res["final_tests"],
                "final_treatments": res["final_treatments"]
            },
            "guideline_citations": {
                "citation": res["literature_citation"],
                "text": res["literature_guideline"]
            },
            "specialist_opinions": opinions_serialized
        }
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return result
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON vital metrics format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consensus error: {str(e)}")

@app.post("/twin/simulate")
async def simulate_patient_twin(
    patient_vitals_json: str = Form(...), # Baseline weight, BP, glucose, HR, etc.
    what_if_scenario_json: str = Form(...) # Target overrides (weight_change, metformin_mg, etc.)
):
    """
    Executes patient digital twin projections under specified intervention scenarios.
    """
    REQUEST_COUNT.labels(endpoint="/twin/simulate").inc()
    start_time = time.perf_counter()
    
    if twin_engine is None:
        raise HTTPException(status_code=503, detail="Digital twin engine offline")
        
    try:
        vitals_dict = json.loads(patient_vitals_json)
        scenario_dict = json.loads(what_if_scenario_json)
        
        # Execute Bayesian Rollout simulation
        simulation_results = twin_engine.run_what_if_simulation(
            patient_vitals=vitals_dict,
            what_if_scenario=scenario_dict,
            steps=12
        )
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return simulation_results
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON parameter formatting")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Digital Twin simulation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
