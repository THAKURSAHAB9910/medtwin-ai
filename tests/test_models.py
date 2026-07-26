import pytest
import torch
from src.models.med_vit import MedicalVisualBranch
from src.models.clinical_vlm import ClinicalVLM
from src.models.lab_encoder import StructuredLabEncoder
from src.models.fusion import TriModalGatedFusion

def test_medical_visual_branch():
    model = MedicalVisualBranch(backbone_name="resnet18", pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    logit, mask, feats = model(x)
    
    assert logit.shape == (2, 1)
    assert mask.shape == (2, 1, 256, 256)
    assert feats.shape[0] == 2
    assert feats.shape[1] == 512 # resnet18 channels

def test_clinical_vlm():
    vlm = ClinicalVLM(use_mock=True, embed_dim=128)
    texts = ["Cough and high fever.", "Stable respiration."]
    
    features, mask = vlm(texts)
    
    assert features.shape == (2, 64, 128)
    assert mask.shape == (2, 64)

def test_structured_lab_encoder():
    encoder = StructuredLabEncoder(input_dim=5, embed_dim=128)
    mock_labs = torch.randn(2, 5) # [B, 5]
    token = encoder(mock_labs)
    
    assert token.shape == (2, 1, 128)

def test_tri_modal_gated_fusion():
    fusion = TriModalGatedFusion(visual_dim=512, semantic_dim=128)
    
    vis = torch.randn(2, 512, 8, 8)
    txt = torch.randn(2, 64, 128)
    lab = torch.randn(2, 1, 128)
    mask = torch.ones(2, 64)
    
    logits, gate = fusion(vis, txt, lab, mask)
    
    assert logits.shape == (2, 1)
    assert gate.shape == (2, 1)
    assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)
