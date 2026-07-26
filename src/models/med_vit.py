import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple, Any

class MedicalVisualBranch(nn.Module):
    """
    Medical Image Branch for MedTwin AI.
    Processes clinical radiographs (e.g. chest X-rays) using a timm backbone.
    Predicts localized lung lesion areas and extracts high-dimensional visual descriptors.
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = False):
        super().__init__()
        # Load backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        feature_info = self.backbone.feature_info.channels()
        c1, c2, c3, c4 = feature_info
        
        # Spatial Decoder for pathology lesion localization
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(c1, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1) # Logit output for lesion mask
        )
        
        # Global visual classification path
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input scan image [B, 3, H, W]
        Returns:
            global_vis_logit: Classification logit [B, 1]
            lesion_map: Dense pathology map [B, 1, H, W]
            vis_features: Feature tensor for multimodal fusion [B, C, H_feat, W_feat]
        """
        feats = self.backbone(x)
        f1, f2, f3, f4 = feats
        
        # Upsampling and skip connections
        x_up = F.relu(self.up4(f4) + f3)
        x_up = F.relu(self.up3(x_up) + f2)
        x_up = F.relu(self.up2(x_up) + f1)
        
        lesion_map = self.segmentation_head(x_up)
        
        pooled = self.global_pool(f4).view(f4.size(0), -1)
        global_vis_logit = self.cls_head(pooled)
        
        return global_vis_logit, lesion_map, f4

if __name__ == "__main__":
    model = MedicalVisualBranch(pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    logit, mask, feats = model(x)
    print("Medical Visual Branch dimensions:")
    print(f" - Logit: {logit.shape}")
    print(f" - Lesion Mask: {mask.shape}")
    print(f" - Visual Features: {feats.shape}")
