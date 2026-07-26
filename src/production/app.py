import io
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
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
from src.vlm.unified_api import MedicalVLMRegistry
from src.vlm.benchmark import VLMPerformanceBenchmarker
from src.nlp.embeddings import MedicalEmbeddingGenerator
from src.nlp.vector_db import LocalVectorDB
from src.nlp.entity_extractor import ClinicalNER
from src.nlp.timeline import ClinicalTimelineCompiler
from src.nlp.rag_engine import ClinicalRAGEngine
from src.intelligence.clinical_fuser import MultimodalIntelligenceFuser
from src.intelligence.reasoning_engine import MultimodalReasoningEngine
from src.imaging.pipeline import MedicalImagingPipeline
from src.imaging.benchmarker import ImagingArchitectureBenchmarker
from src.ocr.pipeline import MedicalOCRPipeline
from src.ocr.benchmarker import OCRArchitectureBenchmarker
from src.synthetic.engine import SyntheticMedicalDataEngine
from src.synthetic.evaluator import SyntheticDataEvaluator

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
vlm_registry: Optional[MedicalVLMRegistry] = None
vlm_benchmarker: Optional[VLMPerformanceBenchmarker] = None
nlp_embedder: Optional[MedicalEmbeddingGenerator] = None
nlp_vector_db: Optional[LocalVectorDB] = None
nlp_ner: Optional[ClinicalNER] = None
nlp_timeline_compiler: Optional[ClinicalTimelineCompiler] = None
nlp_rag_engine: Optional[ClinicalRAGEngine] = None
intel_fuser: Optional[MultimodalIntelligenceFuser] = None
intel_reasoning_engine: Optional[MultimodalReasoningEngine] = None
imaging_pipeline: Optional[MedicalImagingPipeline] = None
imaging_benchmarker: Optional[ImagingArchitectureBenchmarker] = None
ocr_pipeline: Optional[MedicalOCRPipeline] = None
ocr_benchmarker: Optional[OCRArchitectureBenchmarker] = None
synthetic_engine: Optional[SyntheticMedicalDataEngine] = None
synthetic_evaluator: Optional[SyntheticDataEvaluator] = None
img_transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

@app.on_event("startup")
def load_model():
    global model, self_healing_engine, coordinator_agent, twin_engine, vlm_registry, vlm_benchmarker
    global nlp_embedder, nlp_vector_db, nlp_ner, nlp_timeline_compiler, nlp_rag_engine
    global intel_fuser, intel_reasoning_engine
    global imaging_pipeline, imaging_benchmarker
    global ocr_pipeline, ocr_benchmarker
    global synthetic_engine, synthetic_evaluator
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
    vlm_registry = MedicalVLMRegistry(use_mock=True)
    vlm_benchmarker = VLMPerformanceBenchmarker(vlm_registry)
    
    # Initialize Clinical NLP and RAG
    nlp_embedder = MedicalEmbeddingGenerator()
    nlp_vector_db = LocalVectorDB(nlp_embedder)
    nlp_ner = ClinicalNER()
    nlp_timeline_compiler = ClinicalTimelineCompiler()
    nlp_rag_engine = ClinicalRAGEngine(nlp_vector_db, nlp_ner)
    
    # Initialize Clinical Intelligence
    intel_fuser = MultimodalIntelligenceFuser()
    intel_reasoning_engine = MultimodalReasoningEngine()
    
    # Initialize Medical Imaging
    imaging_pipeline = MedicalImagingPipeline()
    imaging_benchmarker = ImagingArchitectureBenchmarker()
    
    # Initialize Medical OCR
    ocr_pipeline = MedicalOCRPipeline(nlp_ner)
    ocr_benchmarker = OCRArchitectureBenchmarker()
    
    # Initialize Synthetic Data Engine
    synthetic_engine = SyntheticMedicalDataEngine()
    synthetic_evaluator = SyntheticDataEvaluator()
    
    nlp_vector_db.add_documents([
        ("American Thoracic Society (ATS) Pneumonia Guideline: Primary recommendation is Ceftriaxone 1g IV daily plus Azithromycin 500mg PO daily.", {"source": "ATS Pneumonia 2019"}),
        ("American Diabetes Association (ADA) Care Standard: Initiate Metformin 500mg PO twice daily as baseline therapy for diabetes mellitus.", {"source": "ADA Diabetes 2024"})
    ])
    print("Models, calibrator, multi-agent coordinator, digital twin, VLM registry, Clinical NLP, Intelligence, Medical Imaging, and OCR modules initialized.")

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

@app.post("/vlm/infer")
async def infer_vlm(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    prompt: str = Form(...),
    task: str = Form("diagnosis")
):
    """
    Unified VLM inference endpoint. Switches model dynamically and executes inference.
    """
    REQUEST_COUNT.labels(endpoint="/vlm/infer").inc()
    start_time = time.perf_counter()
    
    if vlm_registry is None:
        raise HTTPException(status_code=503, detail="VLM registry offline")
        
    try:
        # Load image from file
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Switch model
        vlm_registry.switch_model(model_name)
        
        # Run inference
        outputs = vlm_registry.infer(pil_img, prompt, task)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return outputs
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM inference error: {str(e)}")

@app.post("/vlm/benchmark")
async def benchmark_vlms():
    """
    Runs VLM performance evaluations comparing Qwen2-VL, BioMedCLIP, Florence-2, LayoutLMv3.
    """
    REQUEST_COUNT.labels(endpoint="/vlm/benchmark").inc()
    start_time = time.perf_counter()
    
    if vlm_benchmarker is None:
        raise HTTPException(status_code=503, detail="VLM benchmarker offline")
        
    try:
        results = vlm_benchmarker.benchmark_all(dataloader=None)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VLM benchmark execution error: {str(e)}")

# Fine-tuning job state tracking variables
TRAINING_JOB = {
    "status": "idle",
    "epoch": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "f1": 0.0,
    "error": None
}

def run_background_finetuning():
    global TRAINING_JOB
    try:
        TRAINING_JOB["status"] = "running"
        TRAINING_JOB["epoch"] = 0
        TRAINING_JOB["error"] = None
        
        from src.data.generator import PatientCraftEngine
        from src.finetuning.dataset import ClinicalSFTDataset, MockTokenizer, create_sft_dataloader
        from src.finetuning.trainer import ClinicalPEFTTrainer
        
        engine = PatientCraftEngine(seed=45)
        train_records = [engine.generate_patient_record(has_pneumonia=(i % 2 == 0)) for i in range(12)]
        val_records = [engine.generate_patient_record(has_pneumonia=(i % 2 != 0)) for i in range(4)]
        tokenizer = MockTokenizer()
        
        train_loader = create_sft_dataloader(train_records, tokenizer, batch_size=4)
        val_loader = create_sft_dataloader(val_records, tokenizer, batch_size=4)
        
        class SFTMock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 2)
            def forward(self, x):
                return self.linear(x)
                
        mock_m = SFTMock()
        trainer = ClinicalPEFTTrainer(
            model=mock_m,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir="checkpoints/production"
        )
        
        for epoch in range(3):
            time.sleep(0.5)
            res = trainer.train_epoch(config_type="lora")
            TRAINING_JOB["epoch"] = res["epoch"]
            TRAINING_JOB["loss"] = res["avg_loss"]
            TRAINING_JOB["accuracy"] = res["validation_accuracy"]
            TRAINING_JOB["f1"] = res["validation_f1"]
            
        TRAINING_JOB["status"] = "completed"
    except Exception as e:
        TRAINING_JOB["status"] = "failed"
        TRAINING_JOB["error"] = str(e)

@app.post("/finetune/run")
def run_finetune_job(background_tasks: BackgroundTasks):
    """
    Triggers a Supervised Fine-Tuning (SFT) training loop in the background.
    """
    REQUEST_COUNT.labels(endpoint="/finetune/run").inc()
    if TRAINING_JOB["status"] == "running":
        return {"message": "Fine-tuning job is already running", "status": "running"}
    background_tasks.add_task(run_background_finetuning)
    return {"message": "Fine-tuning job started", "status": "running"}

@app.get("/finetune/status")
def get_finetune_status():
    """
    Queries status of active adapter training job.
    """
    REQUEST_COUNT.labels(endpoint="/finetune/status").inc()
    return TRAINING_JOB

@app.post("/nlp/query")
def query_nlp_rag(query: str = Form(...)):
    """
    Semantic RAG search over medical guidelines, evaluating faithfulness and hallucination rates.
    """
    REQUEST_COUNT.labels(endpoint="/nlp/query").inc()
    start_time = time.perf_counter()
    
    if nlp_rag_engine is None:
        raise HTTPException(status_code=503, detail="NLP RAG engine offline")
        
    try:
        res = nlp_rag_engine.query_clinical_rag(query)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP RAG query error: {str(e)}")

@app.post("/nlp/timeline")
def compile_patient_timeline(events_json: str = Form(...)):
    """
    Chronologically sorts patient health events and compiles longitudinal summaries.
    """
    REQUEST_COUNT.labels(endpoint="/nlp/timeline").inc()
    start_time = time.perf_counter()
    
    if nlp_timeline_compiler is None:
        raise HTTPException(status_code=503, detail="NLP timeline compiler offline")
        
    try:
        events = json.loads(events_json)
        sorted_events = nlp_timeline_compiler.compile_timeline(events)
        narrative = nlp_timeline_compiler.generate_narrative_timeline(events)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return {
            "chronological_events": sorted_events,
            "narrative_timeline": narrative
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON events list")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Timeline compilation error: {str(e)}")

@app.post("/intelligence/diagnose")
async def diagnose_multimodal_patient(
    clinical_note: str = Form(...),
    lab_metrics_json: str = Form(...),
    timeline_events_json: str = Form(...),
    genomic_data_json: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Multimodal clinical intelligence fuser and reasoning diagnostic pipeline.
    """
    REQUEST_COUNT.labels(endpoint="/intelligence/diagnose").inc()
    start_time = time.perf_counter()
    
    if intel_fuser is None or intel_reasoning_engine is None or nlp_timeline_compiler is None:
        raise HTTPException(status_code=503, detail="Clinical intelligence system offline")
        
    try:
        # Load image if present
        pil_img = None
        if file is not None:
            contents = await file.read()
            pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
            
        # Parse inputs
        vitals = json.loads(lab_metrics_json)
        events = json.loads(timeline_events_json)
        genomics = json.loads(genomic_data_json) if genomic_data_json else None
        
        # Compile timeline narrative
        timeline_narrative = nlp_timeline_compiler.generate_narrative_timeline(events)
        
        # Fuse modalities
        fused_state = intel_fuser.fuse_patient_profile(
            image=pil_img,
            clinical_note=clinical_note,
            vitals=vitals,
            timeline_narrative=timeline_narrative,
            genomic_data=genomics
        )
        
        # Run reasoning evaluation
        results = intel_reasoning_engine.evaluate_profile(fused_state)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return results
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data structure passed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multimodal diagnostic error: {str(e)}")

@app.post("/agents/consensus")
def run_agent_board_consensus(
    clinical_note: str = Form(...),
    lab_metrics_json: str = Form(...)
):
    """
    Executes independent multi-specialist diagnostic evaluations,
    returning coordinator consensus alignment and disagreement logs.
    """
    REQUEST_COUNT.labels(endpoint="/agents/consensus").inc()
    start_time = time.perf_counter()
    
    if coordinator_agent is None:
        raise HTTPException(status_code=503, detail="Coordinator agent offline")
        
    try:
        vitals = json.loads(lab_metrics_json)
        patient_data = {
            "clinical_note": clinical_note,
            "lab_metrics": vitals
        }
        
        # Run consensus board
        state = coordinator_agent.run_consensus(patient_data)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return state.coordinator_consensus
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid lab metrics JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-agent consensus failure: {str(e)}")

@app.post("/imaging/analyze")
async def analyze_medical_image(
    modality: str = Form(...),
    task: str = Form(...), # classify, segment, detect, localize, severity
    file: UploadFile = File(...)
):
    """
    Unified route executing specialized medical vision tasks on scans.
    """
    REQUEST_COUNT.labels(endpoint="/imaging/analyze").inc()
    start_time = time.perf_counter()
    
    if imaging_pipeline is None:
        raise HTTPException(status_code=503, detail="Imaging research pipeline offline")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Route to appropriate task
        t_name = task.lower().strip()
        if t_name == "classify":
            res = imaging_pipeline.classify(pil_img, modality)
        elif t_name == "segment":
            res = imaging_pipeline.segment(pil_img, modality)
        elif t_name == "detect":
            res = imaging_pipeline.detect(pil_img, modality)
        elif t_name == "localize":
            res = imaging_pipeline.localize(pil_img, modality)
        elif t_name == "severity":
            res = imaging_pipeline.estimate_severity(pil_img, modality)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported vision task: {task}")
            
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Imaging pipeline processing error: {str(e)}")

@app.post("/ocr/analyze")
async def analyze_medical_document(
    document_type: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Scans patient document files and extracts named clinical entities.
    """
    REQUEST_COUNT.labels(endpoint="/ocr/analyze").inc()
    start_time = time.perf_counter()
    
    if ocr_pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline offline")
        
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Extract text & entities
        res = ocr_pipeline.extract_text_and_entities(pil_img, document_type)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

@app.post("/synthetic/generate")
def generate_synthetic_profile(
    modality: str = Form(...),
    condition: str = Form(...),
    quality: str = Form(...) # high, low-quality, blurred, noisy
):
    """
    Generates synthetic medical scan images and clinical documents matching conditions.
    """
    REQUEST_COUNT.labels(endpoint="/synthetic/generate").inc()
    start_time = time.perf_counter()
    
    if synthetic_engine is None:
        raise HTTPException(status_code=503, detail="Synthetic engine offline")
        
    try:
        # Generate image
        pil_img = synthetic_engine.generate_image(modality, condition, quality)
        
        # Save image to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        import base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Generate documents
        prescription = synthetic_engine.generate_prescription(condition)
        summary = synthetic_engine.generate_discharge_summary(condition)
        labs = synthetic_engine.generate_lab_report(condition)
        history = synthetic_engine.generate_patient_history(condition)
        
        latency = time.perf_counter() - start_time
        LATENCY_SUMMARY.observe(latency)
        
        return {
            "condition": condition,
            "modality": modality,
            "quality": quality,
            "synthetic_image_b64": img_str,
            "synthetic_prescription": prescription,
            "synthetic_discharge_summary": summary,
            "synthetic_lab_report": labs,
            "synthetic_patient_history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthetic generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
