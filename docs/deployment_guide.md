# MedTwin AI: Production Deployment & Operations Guide

This guide details how to deploy, monitor, and scale the **MedTwin AI** clinical decision support REST server.

---

## 1. Containerization (Docker)

To deploy the REST server in a staging or production container:

### 1.1 Build the Image
```bash
docker build -t medtwin-api:latest .
```

### 1.2 Run the Service
Expose the service on port 8000:
```bash
docker run -d -p 8000:8000 --name medtwin-service medtwin-api:latest
```

---

## 2. API Endpoints Reference

### 2.1 Health Probe
* **Endpoint**: `/health`
* **Method**: `GET`
* **Response**: `{"status": "healthy", "clinical_modules_ready": true}`

### 2.2 Clinical Case Verification
* **Endpoint**: `/predict`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Parameters**:
  - `file`: The radiograph scan image file (JPEG, PNG).
  - `clinical_note`: Text transcript of physical symptoms.
  - `lab_metrics_json`: Serialized JSON vitals matching:
    ```json
    {"age": 65, "temperature": 102.5, "heart_rate": 110, "wbc_count": 15000, "oxygen_saturation": 89}
    ```

#### Python Request Example
```python
import json
import requests

url = "http://127.0.0.1:8000/predict"

vitals = {
    "age": 65.0,
    "temperature": 102.5,
    "heart_rate": 110.0,
    "wbc_count": 15000.0,
    "oxygen_saturation": 89.0
}

files = {
    "file": ("radiograph.png", open("chest_scan.png", "rb"), "image/png")
}
data = {
    "clinical_note": "Patient presents with cough and dyspnea. Radiograph indicates focal lung opacities.",
    "lab_metrics_json": json.dumps(vitals)
}

res = requests.post(url, files=files, data=data)
print(res.json())
```

#### Output Response Example
```json
{
  "diagnostic_metrics": {
    "base_calibrated_prob": 0.4735,
    "healed_calibrated_prob": 1.0,
    "prediction_set": [1],
    "diagnosis_label": "Pneumonia",
    "is_active_disease": true,
    "requires_clinical_audit": false
  },
  "self_healing_trace": {
    "action_taken": "healed_infection_verification",
    "reasoning_log": "Diagnostic prediction was clinically ambiguous. Triggering self-healing... High physiological or visual disease markers detected: Temp: 102.5F, WBC: 15000/uL, SpO2: 89.0%, pathology density: 0.1245. Overridden to Pneumonia.",
    "physiological_audit": {
      "fever": true,
      "leukocytosis": true,
      "hypoxemia": true,
      "has_infection_evidence": true,
      "has_healthy_evidence": false,
      "details": "Temp: 102.5F, WBC: 15000.0/uL, SpO2: 89.0%"
    },
    "visual_lesion_audit": {
      "mean_pathology_score": 0.1245,
      "has_visual_lesion": false
    }
  }
}
```

---

## 3. Operations Monitoring (Prometheus)

Exposes raw scrapers on `/metrics`.

### 3.1 Telemetry Counters
1. `medtwin_requests_total`: Tracks API hits per route.
2. `medtwin_latency_seconds`: Summary tracking inference timings.
3. `medtwin_self_healing_total`: Tracks triggers grouped by action (`healed_normal_verification`, `healed_infection_verification`, `routed_to_physician`).

### 3.2 Prometheus Config
```yaml
scrape_configs:
  - job_name: 'medtwin'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:8000']
```
