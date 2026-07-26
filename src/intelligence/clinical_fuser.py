import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Any, List, Optional

class MultimodalIntelligenceFuser:
    """
    Fuses visual, textual, tabular, chronological, and genomic modalities
    into a unified aligned clinical state representation.
    """
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        # Projection layer for genomic variables
        self.genomic_proj = nn.Linear(5, embedding_dim)

    def fuse_patient_profile(
        self,
        image: Optional[Image.Image],
        clinical_note: str,
        vitals: Dict[str, float],
        timeline_narrative: str,
        genomic_data: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Fuses all patient modalities into a structured representation.
        """
        # 1. Text alignment (notes + chronological history narrative)
        fused_narrative = f"CLINICAL HISTORY: {clinical_note}\nLONGITUDINAL PROGRESSION: {timeline_narrative}"
        
        # 2. Vitals alignment
        aligned_vitals = {
            "age": float(vitals.get("age", 50)),
            "temperature": float(vitals.get("temperature", 98.6)),
            "heart_rate": float(vitals.get("heart_rate", 80)),
            "wbc_count": float(vitals.get("wbc_count", 7500)),
            "oxygen_saturation": float(vitals.get("oxygen_saturation", 97)),
            "blood_pressure": float(vitals.get("blood_pressure", 120)),
            "blood_glucose": float(vitals.get("blood_glucose", 100)),
            "weight": float(vitals.get("weight", 80.0))
        }
        
        # 3. Genomics alignment
        aligned_genomics = {
            "EGFR_mutation": "not_detected",
            "BRCA1_mutation": "not_detected",
            "HLA_DRB1_allele": "negative"
        }
        if genomic_data:
            for k, v in genomic_data.items():
                if k in aligned_genomics:
                    aligned_genomics[k] = v

        # 4. Compile unified state
        fused_state = {
            "narrative_text": fused_narrative,
            "vitals": aligned_vitals,
            "genomics": aligned_genomics,
            "has_radiograph": image is not None
        }
        
        return fused_state
