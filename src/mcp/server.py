import json
from typing import Dict, Any, List

class MCPServer:
    """
    Model Context Protocol (MCP) Server simulation.
    Exposes healthcare database endpoints as JSON-RPC 2.0 tools.
    """
    def __init__(self):
        # Seed mock patient databases to ensure clinical context consistency
        self.patient_db = {
            "PT-8891": {
                "name": "Jane Doe",
                "dob": "1978-05-14",
                "gender": "female",
                "fhir_patient": {
                    "resourceType": "Patient",
                    "id": "PT-8891",
                    "name": [{"family": "Doe", "given": ["Jane"]}],
                    "gender": "female",
                    "birthDate": "1978-05-14"
                },
                "hl7_demographics": "MSH|^~\\&|EHR|HOSP|||2026073020||ADT^A08|MSG001|P|2.5\rEVN||2026073020\rPID|||PT-8891||Doe^Jane||19780514|F",
                "ehr_notes": "Patient presents with persistent dry cough, moderate dyspnea, and low-grade pyrexia. Active history of chronic mild asthma managed with Albuterol inhaler.",
                "lab_results": {
                    "temperature": 101.5,
                    "wbc_count": 14200.0,
                    "oxygen_saturation": 91.0,
                    "blood_pressure": 130.0,
                    "heart_rate": 92.0,
                    "observation_fhir": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
                        "valueQuantity": {"value": 92.0, "unit": "bpm"}
                    }
                },
                "pacs_dicom": {
                    "study_id": "STUDY-XRAY-44",
                    "modality": "CR",
                    "description": "Chest Radiograph PA/Lateral view",
                    "findings": "Bilateral patchy alveolar infiltrates consistent with pulmonary consolidation, primarily in the lower right lung segment. Bounding margins indicate active inflammation."
                },
                "pharmacy": {
                    "active_medications": ["Albuterol HFA Inhaler 90mcg (1-2 puffs q4-6h prn)"],
                    "contraindications": ["Metformin + Severe Hypoxia risk", "Beta-blocker + Active Asthma risk"]
                },
                "hospital_management": {
                    "admission_status": "Observation",
                    "attending_physician": "Dr. Sarah Carter",
                    "department": "Pulmonary Medicine",
                    "room": "Bed 4B",
                    "appointment_history": ["2026-03-12 (Routine Followup)", "2026-07-30 (Emergency Presentation)"]
                },
                "insurance": {
                    "provider": "Blue Cross Shield",
                    "policy_number": "BCS-990123",
                    "coverage_limit": 25000.0,
                    "pre_auth_required": True,
                    "eligibility_status": "Active"
                },
                "gdrive_notes": "Internal clinical file: Jane Doe has shown elevated sensitivity to dust mites. Advise environmental adjustments.",
                "s3_archive": {
                    "archive_id": "S3-ARC-Doe-2026",
                    "bucket": "medtwin-clinical-archives",
                    "key": "PT-8891/2026_discharge_summary.pdf"
                }
            }
        }
        
        self.guidelines_db = {
            "pneumonia": "ATS/IDSA Guidelines (2019): Empiric therapy for community-acquired pneumonia (CAP) in outpatients without comorbidities: Amoxicillin or Doxycycline. With comorbidities: Respiratory fluoroquinolone (e.g. Levofloxacin) OR combination Beta-lactam (Ceftriaxone) plus Macrolide (Azithromycin). Monitor SpO2 levels and avoid drug contraindications.",
            "diabetes": "ADA Standards of Medical Care (2024): Metformin is the preferred initial pharmacologic agent for the treatment of type 2 diabetes. T2DM with cardiovascular disease history warrants GLP-1 receptor agonists or SGLT2 inhibitors."
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return the JSON schemas of all tools exposed by the MCP Server."""
        return [
            {
                "name": "get_ehr_record",
                "description": "Retrieves Patient Demographics, FHIR patient resources, and active doctor notes from EHR.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique medical record identifier (e.g. PT-8891)"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_laboratory_results",
                "description": "Retrieves complete panel observations, temperatures, leukocyte count, and oxygen saturation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_radiology_pacs",
                "description": "Retrieves medical scan DICOM details and radiologist findings from PACS repository.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_pharmacy_records",
                "description": "Retrieves patient drug list and alerts for active pharmacological contraindications.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_hospital_management",
                "description": "Queries attending physician, admission room locations, and appointment history.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_insurance_status",
                "description": "Retrieves pre-authorization details, policies, and copay bounds.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            },
            {
                "name": "get_clinical_guidelines",
                "description": "Retrieves medical knowledge recommendations for a target pathology (e.g. pneumonia).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pathology": {"type": "string", "description": "Target disease name"}
                    },
                    "required": ["pathology"]
                }
            },
            {
                "name": "get_cloud_archives",
                "description": "Queries patient AWS S3 archives and Google Drive physician notes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "string", "description": "Unique patient ID"}
                    },
                    "required": ["patient_id"]
                }
            }
        ]

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute JSON-RPC tool resolution.
        """
        # Validate patient existence for patient-specific queries
        patient_id = arguments.get("patient_id")
        if name != "get_clinical_guidelines":
            if not patient_id or patient_id not in self.patient_db:
                return {
                    "error": {
                        "code": -32602,
                        "message": f"Patient ID '{patient_id}' not found in registry."
                    }
                }

        record = self.patient_db.get(patient_id) if patient_id else None

        try:
            if name == "get_ehr_record":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "fhir_patient": record["fhir_patient"],
                        "hl7_demographics": record["hl7_demographics"],
                        "clinical_notes": record["ehr_notes"]
                    }
                }
            elif name == "get_laboratory_results":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "vitals": {
                            "temperature": record["lab_results"]["temperature"],
                            "wbc_count": record["lab_results"]["wbc_count"],
                            "oxygen_saturation": record["lab_results"]["oxygen_saturation"],
                            "blood_pressure": record["lab_results"]["blood_pressure"],
                            "heart_rate": record["lab_results"]["heart_rate"]
                        },
                        "fhir_observation": record["lab_results"]["observation_fhir"]
                    }
                }
            elif name == "get_radiology_pacs":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "pacs_study": record["pacs_dicom"]
                    }
                }
            elif name == "get_pharmacy_records":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "active_medications": record["pharmacy"]["active_medications"],
                        "contraindications": record["pharmacy"]["contraindications"]
                    }
                }
            elif name == "get_hospital_management":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "admission_status": record["hospital_management"]["admission_status"],
                        "room": record["hospital_management"]["room"],
                        "attending_physician": record["hospital_management"]["attending_physician"],
                        "appointment_history": record["hospital_management"]["appointment_history"]
                    }
                }
            elif name == "get_insurance_status":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "insurance_status": record["insurance"]
                    }
                }
            elif name == "get_clinical_guidelines":
                pathology = arguments.get("pathology", "").lower()
                guideline = self.guidelines_db.get(pathology, "No guideline matched.")
                return {
                    "result": {
                        "pathology": pathology,
                        "guideline": guideline
                    }
                }
            elif name == "get_cloud_archives":
                return {
                    "result": {
                        "patient_id": patient_id,
                        "gdrive_notes": record["gdrive_notes"],
                        "s3_archive": record["s3_archive"]
                    }
                }
            else:
                return {
                    "error": {
                        "code": -32601,
                        "message": f"Tool '{name}' not found."
                    }
                }
        except Exception as e:
            return {
                "error": {
                    "code": -32603,
                    "message": f"Internal tool execution error: {str(e)}"
                }
            }
