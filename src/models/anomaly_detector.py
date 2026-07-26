import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Tuple, Any

class VisualAnomalyDetector(nn.Module):
    """
    Visual branch of UniDocShield.
    Extracts high-resolution texture features from the document image,
    predicting both pixel-level tampering regions (segmentation) and 
    global document authenticity (classification).
    Uses a timm backbone for multi-scale feature extraction.
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        # Create backbone as feature extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4) # Extract features at multiple resolutions (e.g., 1/4, 1/8, 1/16, 1/32)
        )
        
        # Get feature channels
        feature_info = self.backbone.feature_info.channels()
        
        # Feature sizes for resnet18: e.g. [64, 128, 256, 512]
        c1, c2, c3, c4 = feature_info
        
        # Segmentation Decoder: Simple FPN/U-Net style fusion to reconstruct the anomaly mask
        # Downsample factor of final mask is 1x (original image resolution)
        self.up4 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        
        # Final reconstruction layer to get [1, H, W] anomaly map
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(c1, 32, kernel_size=2, stride=2), # up to 1/2
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), # up to 1/1
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )
        
        # Global Classifier Branch: Pools the deepest features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.class_head = nn.Sequential(
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Binary logit output
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image tensor of shape [B, 3, H, W]
        Returns:
            global_logit: Forgery classification logits [B, 1]
            anomaly_map: Pixel-level logits [B, 1, H, W]
            features: Deepest feature maps for cross-attention fusion [B, C, H_feat, W_feat]
        """
        # Multi-scale feature extraction
        feats = self.backbone(x)
        f1, f2, f3, f4 = feats
        
        # Decoder forward path with residual skips
        x_up = F.relu(self.up4(f4) + f3)
        x_up = F.relu(self.up3(x_up) + f2)
        x_up = F.relu(self.up2(x_up) + f1)
        
        anomaly_map = self.seg_head(x_up)
        
        # Global classification path
        pooled = self.global_pool(f4).view(f4.size(0), -1)
        global_logit = self.class_head(pooled)
        
        # We also return f4 (deepest visual features) for multi-modal cross-attention fusion
        return global_logit, anomaly_map, f4

if __name__ == "__main__":
    # Validate tensor flows
    model = VisualAnomalyDetector(pretrained=False)
    x = torch.randn(2, 3, 384, 384)
    cls_out, seg_out, feats = model(x)
    print("Visual Anomaly Detector Check:")
    print(f"Global logit shape: {cls_out.shape}")
    print(f"Anomaly map shape: {seg_out.shape}")
    print(f"Visual features shape: {feats.shape}")
