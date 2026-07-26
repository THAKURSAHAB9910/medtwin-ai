import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

from src.models.med_vit import MedicalVisualBranch
from src.models.clinical_vlm import ClinicalVLM
from src.models.lab_encoder import StructuredLabEncoder
from src.models.fusion import TriModalGatedFusion
from src.models.uncertainty import UncertaintyCalibrator

class MedTwinAIModule(pl.LightningModule):
    """
    Main PyTorch Lightning module for MedTwin AI.
    Integrates the tri-modal visual, textual, and structured branches, 
    running multi-task diagnostic classification and lesion segmentation.
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        embed_dim: int = 256,
        use_mock_vlm: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        segmentation_weight: float = 2.0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seg_weight = segmentation_weight
        
        # Instantiate networks
        self.visual_branch = MedicalVisualBranch(backbone_name=backbone_name, pretrained=False)
        self.vlm_branch = ClinicalVLM(use_mock=use_mock_vlm, embed_dim=embed_dim)
        self.lab_branch = StructuredLabEncoder(input_dim=5, embed_dim=embed_dim)
        self.fusion_layer = TriModalGatedFusion(visual_dim=512, semantic_dim=embed_dim)
        
        # Instantiate calibrator
        self.calibrator = UncertaintyCalibrator()
        
        # Validation collectors for calibration fitting
        self.val_logits_epoch = []
        self.val_labels_epoch = []

        # Loss functions
        self.bce_class = nn.BCEWithLogitsLoss()
        self.bce_seg = nn.BCEWithLogitsLoss()

    def forward(self, image: torch.Tensor, notes: list, labs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Tri-Modal Forward Pass.
        """
        # 1. Visual Feature Extraction
        vis_logit, lesion_map, vis_feats = self.visual_branch(image)
        
        # 2. Text Note Feature Extraction
        text_feats, text_mask = self.vlm_branch(notes, device=self.device)
        
        # 3. Structured Vitals Token Extraction
        lab_token = self.lab_branch(labs)
        
        # 4. Gated Tri-modal fusion
        fused_logit, gate_val = self.fusion_layer(vis_feats, text_feats, lab_token, text_mask)
        
        return {
            "fused_logit": fused_logit,
            "lesion_map": lesion_map,
            "vis_logit": vis_logit,
            "gate_val": gate_val
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        mask = batch["mask"]
        notes = batch["clinical_note"]
        labs = batch["lab_metrics"]
        label = batch["label"].float().unsqueeze(1)
        
        outputs = self.forward(image, notes, labs)
        
        # Diagnostic Classification loss
        loss_class = self.bce_class(outputs["fused_logit"], label)
        
        # Lesion Segmentation loss
        loss_seg = self.bce_seg(outputs["lesion_map"], mask)
        
        # Total loss
        total_loss = loss_class + self.seg_weight * loss_seg
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(image))
        self.log("train_class_loss", loss_class, on_step=False, on_epoch=True, batch_size=len(image))
        self.log("train_seg_loss", loss_seg, on_step=False, on_epoch=True, batch_size=len(image))
        
        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        image = batch["image"]
        mask = batch["mask"]
        notes = batch["clinical_note"]
        labs = batch["lab_metrics"]
        label = batch["label"].float().unsqueeze(1)
        
        outputs = self.forward(image, notes, labs)
        
        loss_class = self.bce_class(outputs["fused_logit"], label)
        loss_seg = self.bce_seg(outputs["lesion_map"], mask)
        val_loss = loss_class + self.seg_weight * loss_seg
        
        self.val_logits_epoch.append(outputs["fused_logit"].detach().cpu())
        self.val_labels_epoch.append(batch["label"].detach().cpu())
        
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(image))
        return val_loss

    def on_validation_epoch_end(self):
        if len(self.val_logits_epoch) == 0:
            return
            
        all_logits = torch.cat(self.val_logits_epoch, dim=0)
        all_labels = torch.cat(self.val_labels_epoch, dim=0)
        
        # Fit Temperature Scaling on validation split
        self.calibrator.fit_temperature(all_logits, all_labels, epochs=50)
        
        # Compute calibrated probabilities
        calibrated_probs = self.calibrator.get_calibrated_probability(all_logits).squeeze(-1).numpy()
        labels_np = all_labels.numpy()
        
        # Compute ECE
        ece = self.calibrator.compute_ece(calibrated_probs, labels_np)
        self.log("val_ece", ece, prog_bar=True)
        
        # Fit Split Conformal Prediction for alpha=0.1
        self.calibrator.fit_conformal(calibrated_probs, labels_np, alpha=0.1)
        
        # Clear validation collectors for next epoch
        self.val_logits_epoch.clear()
        self.val_labels_epoch.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        }
