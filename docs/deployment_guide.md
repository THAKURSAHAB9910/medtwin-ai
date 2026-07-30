# MedTwin AI: Deployment Guide

## 1. Local Server Launch
MedTwin AI runs on a high-throughput FastAPI wrapper server. You can spin up the service locally using Uvicorn:

```bash
python -m uvicorn src.production.app:app --host 127.0.0.1 --port 8000 --workers 4
```

---

## 2. API Endpoint Payload Registry

### 2.1 Medical OCR & NER
* **Endpoint**: `POST /ocr/analyze`
* **Payload**:
  * `document_type`: string (e.g. `prescription`, `lab_report`)
  * `file`: Binary file upload (scanned image)

### 2.2 Patient Digital Twin
* **Endpoint**: `POST /digital_twin/update`
* **Payload**:
  * `patient_id`: string
  * `vital_name`: string
  * `vital_value`: float

### 2.3 Synthetic Generation
* **Endpoint**: `POST /synthetic/generate`
* **Payload**:
  * `modality`: string (e.g. `x-ray`, `mri`)
  * `condition`: string (e.g. `alkaptonuria`, `glioblastoma`)
  * `quality`: string (e.g. `high`, `blurred`, `noisy`)

---

## 3. Monitoring & Telemetry
The application exports real-time service metrics to Prometheus:
* **Metrics URL**: `http://127.0.0.1:8000/metrics`
* **Metrics Exposed**:
  * `medtwin_requests_total`: Tracks api request counts per endpoint.
  * `medtwin_latency_seconds`: Measures exact model prediction latency.
  * `medtwin_self_healing_total`: Counts active pipeline healing occurrences.

---

## 4. Enterprise Database Schema & Redis Sessions

### 4.1 Redis Caching & Session Configuration
For session management and agent state caching, MedTwin AI connects to Redis. Configure the connection parameters in environment variables:
* `REDIS_HOST`: Localhost / container service link.
* `REDIS_PORT`: Default `6379`.
* `REDIS_DB`: Database index for sessions (defaults to `0`).

### 4.2 FHIR Demographics Schema (Pydantic / SQL Model)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "FHIRPatient",
  "type": "object",
  "properties": {
    "resourceType": { "type": "string", "const": "Patient" },
    "id": { "type": "string" },
    "active": { "type": "boolean" },
    "name": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "family": { "type": "string" },
          "given": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "gender": { "type": "string", "enum": ["male", "female", "other", "unknown"] },
    "birthDate": { "type": "string", "format": "date" }
  },
  "required": ["resourceType", "id", "name", "gender", "birthDate"]
}
```

---

## 5. Multi-Agent & n8n Webhook Architecture
After human-in-the-loop review, the system triggers the post-diagnosis pipeline via webhooks:
* **Webhook Endpoint**: `POST /n8n/webhook`
* **Payload Registry**:
```json
{
  "event": "clinical_diagnosis_completed",
  "patient_id": "PT-8891",
  "diagnosis": "Community-Acquired Pneumonia",
  "is_high_risk": true,
  "report_details": {
    "prescribed_treatment": [
      "Initiate Ceftriaxone 1g IV daily",
      "Initiate Azithromycin 500mg PO daily"
    ]
  }
}
```
* **High-Risk Escalations**: If `is_high_risk` is `true`, n8n automatically flags the case for Emergency team dispatch, priority SMS alerts, and CMO audit logs write-back.
