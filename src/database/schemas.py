import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Pydantic Schemas modeling HL7 and FHIR resources
class FHIRPatient(BaseModel):
    resourceType: str = "Patient"
    id: str
    active: bool = True
    name: List[Dict[str, Any]]
    gender: str
    birthDate: str

class FHIRObservation(BaseModel):
    resourceType: str = "Observation"
    id: str
    status: str = "final"
    code: Dict[str, Any]
    subject: Dict[str, str]  # e.g., {"reference": "Patient/PT-8891"}
    valueQuantity: Dict[str, Any]

class HL7Segment(BaseModel):
    segment_type: str  # PID, MSH, OBR, EVN
    fields: List[str]

class AuditTrailEntry(BaseModel):
    entry_id: str
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    action: str
    patient_id: str
    user_id: str
    details: Dict[str, Any]

class RedisCacheSimulation:
    """
    Redis-compatible key-value cache database simulator.
    Acts as a high-speed workflow state and session cache layer.
    """
    def __init__(self):
        self.store: Dict[str, str] = {}
        self.expiries: Dict[str, float] = {}

    def ping(self) -> bool:
        """Pings the cache server."""
        return True

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set a value in the cache with optional TTL expiration."""
        self.store[key] = value
        if ex:
            self.expiries[key] = time.time() + ex
        else:
            self.expiries.pop(key, None)
        return True

    def get(self, key: str) -> Optional[str]:
        """Retrieve a value if not expired."""
        if key in self.expiries and time.time() > self.expiries[key]:
            self.store.pop(key, None)
            self.expiries.pop(key, None)
            return None
        return self.store.get(key)

    def exists(self, key: str) -> bool:
        """Check key existence."""
        return self.get(key) is not None

    def delete(self, key: str) -> bool:
        """Removes key from cache store."""
        self.store.pop(key, None)
        self.expiries.pop(key, None)
        return True
