import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import Dict, Any, List

class ClinicalExplainer:
    """
    Medical Failure Analysis & Explainability Engine.
    Plots chest radiograph lesion segmentations, generates visual overlays,
    and documents clinical note text mappings alongside vital abnormalities.
    """
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_heatmap_overlay(self, image_np: np.ndarray, lesion_map: np.ndarray) -> np.ndarray:
        """
        Projects predicted lesion maps onto X-ray images as color overlays.
        """
        if image_np.max() <= 1.01:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
            
        h, w = image_np.shape[:2]
        
        # Resize lesion map to match image size
        pil_map = Image.fromarray((lesion_map * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        resized_map = np.array(pil_map) / 255.0
        
        cmap = plt.get_cmap("jet")
        rgba_heatmap = cmap(resized_map)
        rgb_heatmap = (rgba_heatmap[..., :3] * 255).astype(np.uint8)
        
        alpha_map = np.expand_dims(resized_map, axis=-1)
        blended = (1.0 - 0.4 * alpha_map) * image_np + (0.4 * alpha_map) * rgb_heatmap
        return blended.astype(np.uint8)

    def explain_sample(
        self,
        image_tensor: torch.Tensor,
        lesion_tensor: torch.Tensor,
        metadata: List[Dict[str, Any]],
        clinical_note: str,
        save_name: str = "medtwin_clinical_explanation.png"
    ) -> str:
        """
        Saves a 3-panel clinical diagnosis report:
        Panel 1: Chest X-ray with marked clinical annotation boxes
        Panel 2: Pathology Lesion Map
        Panel 3: Heatmap Overlay
        """
        # Convert image tensor back to numpy [H, W, 3] (reverse ImageNet normalization)
        img_np = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = (img_np * std + mean) * 255.0
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Crucial bugfix: detach and squeeze lesion tensor
        lesion_map = torch.sigmoid(lesion_tensor).detach().cpu().numpy().squeeze()
        
        overlay = self.generate_heatmap_overlay(img_np, lesion_map)
        
        # Bbox annotations drawing
        pil_highlight = Image.fromarray(img_np.copy())
        draw = ImageDraw.Draw(pil_highlight)
        
        h_t, w_t = img_np.shape[:2]
        
        for item in metadata:
            marker = item.get("marker", "")
            if marker == "lesion_bbox":
                bbox = item.get("value", [0, 0, 0, 0])
                if sum(bbox) > 0:
                    # Scale coordinates from 512x512 original generator size to image tensor size
                    x1 = int(bbox[0] * w_t / 512)
                    y1 = int(bbox[1] * h_t / 512)
                    x2 = int(bbox[2] * w_t / 512)
                    y2 = int(bbox[3] * h_t / 512)
                    
                    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
                    draw.text((x1 + 2, y1 + 2), "Consolidation Zone", fill=(255, 0, 0))

        # Setup diagnostic subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        axes[0].imshow(pil_highlight)
        axes[0].set_title("Chest Radiograph (Annotation Box)")
        axes[0].axis("off")
        
        axes[1].imshow(lesion_map, cmap="inferno")
        axes[1].set_title("Pathology Lesion Prediction")
        axes[1].axis("off")
        
        axes[2].imshow(overlay)
        axes[2].set_title("Clinical Heatmap Overlay")
        axes[2].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"\nDiagnostic explanation report exported: {save_path}")
        print("Clinical Interpretation Summary:")
        print(f" - Physician Transcription: \"{clinical_note}\"")
        for item in metadata:
            marker = item.get("marker", "")
            if marker != "lesion_bbox":
                val = item.get("value")
                thr = item.get("critical_threshold")
                op = item.get("operator")
                status = "CRITICAL" if (op == "greater" and val > thr) or (op == "less" and val < thr) else "NORMAL"
                print(f" - Metric '{marker:18}': {val:10} | Threshold: {thr:6} | Status: {status}")
                
        return save_path

if __name__ == "__main__":
    explainer = ClinicalExplainer(output_dir=".")
    
    # Mock inputs
    img_t = torch.randn(3, 384, 384)
    img_t = (img_t - img_t.min()) / (img_t.max() - img_t.min())
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_t = (img_t - mean) / std
    
    lesion_t = torch.randn(1, 384, 384)
    meta = [
        {"marker": "lesion_bbox", "value": [100, 150, 250, 300]},
        {"marker": "temperature", "value": 102.5, "critical_threshold": 100.4, "operator": "greater"}
    ]
    note = "Severe fever. Right-sided haziness observed."
    
    p = explainer.explain_sample(img_t, lesion_t, meta, note, save_name="test_med_explanation.png")
    if os.path.exists(p):
        os.remove(p)
        print("Medical Explainer self-test complete.")
