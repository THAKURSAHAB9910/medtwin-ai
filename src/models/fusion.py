import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class TriModalGatedFusion(nn.Module):
    """
    Tri-Modal Gated Cross-Attention Fusion Network.
    Enables medical visual tokens (Query) to reason jointly over clinical notes (Key/Value) 
    and structured vitals (Key/Value) through cross-attention, balanced by an adaptive gate.
    """
    def __init__(self, visual_dim: int = 512, semantic_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        
        # Project visual channel to semantic dim
        self.proj_visual = nn.Linear(visual_dim, semantic_dim)
        
        # Multihead Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=semantic_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Sigmoid gate network based on pooled visual feature maps
        self.gate_net = nn.Sequential(
            nn.Linear(visual_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Normalization and FFN
        self.norm1 = nn.LayerNorm(semantic_dim)
        self.norm2 = nn.LayerNorm(semantic_dim)
        self.ffn = nn.Sequential(
            nn.Linear(semantic_dim, semantic_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(semantic_dim * 2, semantic_dim)
        )
        
        # Unified Tri-modal diagnosis classifier
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(semantic_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1) # Binary diagnosis logit (e.g. 0: Normal, 1: Pneumonia)
        )

    def forward(
        self,
        visual_feats: torch.Tensor,
        text_feats: torch.Tensor,
        lab_token: torch.Tensor,
        text_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_feats: [B, visual_dim, H_feat, W_feat]
            text_feats: [B, seq_len, semantic_dim]
            lab_token: [B, 1, semantic_dim]
            text_mask: [B, seq_len] (1 = valid, 0 = padding)
        Returns:
            diagnosis_logits: Binary disease logits [B, 1]
            gate_val: The visual-semantic gate coefficient [B, 1]
        """
        B, C, H, W = visual_feats.shape
        
        # 1. Flatten visual tokens [B, C, H, W] -> [B, H*W, semantic_dim]
        visual_flat = visual_feats.view(B, C, -1).transpose(1, 2)
        visual_proj = self.proj_visual(visual_flat)
        
        # 2. Concatenate keys and values: text features and lab tokens
        # Combined context: [B, seq_len + 1, semantic_dim]
        keys_values = torch.cat([text_feats, lab_token], dim=1)
        
        # Construct key padding mask
        # lab token is always valid (mask value = 1.0), so we append a True column to text mask
        B_text_seq = text_feats.size(1)
        lab_mask_val = torch.ones((B, 1), device=text_feats.device)
        combined_mask = torch.cat([text_mask, lab_mask_val], dim=1)
        # Convert mask to key padding mask (True indicates padding elements to ignore)
        key_padding_mask = (combined_mask == 0)
        
        # 3. Cross Attention
        attn_out, _ = self.cross_attn(
            query=visual_proj,
            key=keys_values,
            value=keys_values,
            key_padding_mask=key_padding_mask
        )
        
        # LayerNorm and FFN
        x1 = self.norm1(visual_proj + attn_out)
        x2 = self.norm2(x1 + self.ffn(x1))
        
        # 4. Adaptive Gating
        visual_pooled = visual_feats.mean(dim=[2, 3]) # [B, visual_dim]
        gate_val = self.gate_net(visual_pooled) # [B, 1]
        
        gate_broadcast = gate_val.unsqueeze(-1)
        fused = (1.0 - gate_broadcast) * visual_proj + gate_broadcast * x2
        
        # Pool to final diagnosis vector
        fused_pooled = fused.mean(dim=1) # [B, semantic_dim]
        diagnosis_logits = self.diagnosis_classifier(fused_pooled)
        
        return diagnosis_logits, gate_val

if __name__ == "__main__":
    fusion = TriModalGatedFusion()
    vis = torch.randn(2, 512, 12, 12)
    txt = torch.randn(2, 64, 256)
    lab = torch.randn(2, 1, 256)
    mask = torch.ones(2, 64)
    logits, gate = fusion(vis, txt, lab, mask)
    print("Tri-Modal Gated Fusion Dimensions:")
    print(f" - Logits: {logits.shape}")
    print(f" - Gate value: {gate.shape}")
