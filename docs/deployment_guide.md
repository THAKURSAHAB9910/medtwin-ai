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
