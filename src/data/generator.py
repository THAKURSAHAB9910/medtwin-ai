import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import Dict, Any, Tuple, List

class PatientCraftEngine:
    """
    PatientCraft: A medical synthetic generator.
    Simulates patient records including chest radiographs (images), localized lesions (segmentation masks),
    physician transcripts (clinical notes), and vital statistics (tabular metrics).
    """
    def __init__(self, width: int = 512, height: int = 512, seed: int = 42):
        self.width = width
        self.height = height
        random.seed(seed)
        np.random.seed(seed)

    def draw_mock_chest_xray(self, has_pneumonia: bool) -> Tuple[Image.Image, Image.Image, List[int]]:
        """
        Draws a stylized, synthetic chest X-ray image.
        Returns:
            xray_img: Grayscale PIL Image representing a chest X-ray
            mask_img: Binary PIL Image representing localized pneumonia consolidation (lesion mask)
            lesion_bbox: Bounding box coordinates [x1, y1, x2, y2] of the lesion
        """
        # Create dark background representing tissue density
        img = Image.new("RGB", (self.width, self.height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        
        # Draw ribs/spine outline (stylized lung cavities)
        draw.ellipse([(100, 80), (230, 420)], fill=(60, 60, 60)) # Left Lung Cavity
        draw.ellipse([(282, 80), (412, 420)], fill=(60, 60, 60)) # Right Lung Cavity
        
        # Draw central mediastinum (heart/spine white zone)
        draw.polygon([(230, 80), (282, 80), (290, 240), (320, 320), (250, 420), (220, 320)], fill=(120, 120, 120))
        draw.ellipse([(220, 250), (300, 340)], fill=(150, 150, 150)) # Heart outline
        
        # Initialize empty mask
        mask = Image.new("L", (self.width, self.height), color=0)
        draw_mask = ImageDraw.Draw(mask)
        
        lesion_bbox = [0, 0, 0, 0]
        
        if has_pneumonia:
            # Draw a hazy consolidation patch (opacity) in the lower lung zone
            # Pneumonia appears as white/grey opacity overlay on dark lung fields
            lung_side = random.choice(["left", "right"])
            
            if lung_side == "left":
                # Lower left cavity bounds roughly [130, 280, 210, 380]
                cx, cy = random.randint(140, 180), random.randint(290, 340)
                r = random.randint(35, 55)
            else:
                # Lower right cavity bounds roughly [300, 280, 380, 380]
                cx, cy = random.randint(320, 360), random.randint(290, 340)
                r = random.randint(35, 55)
                
            # Bounding box
            lesion_bbox = [cx - r, cy - r, cx + r, cy + r]
            
            # Draw hazy lesion on X-ray
            # Standard PIL doesn't do smooth transparency easily without mask blending
            lesion_overlay = Image.new("RGB", (self.width, self.height), color=(0, 0, 0))
            draw_overlay = ImageDraw.Draw(lesion_overlay)
            draw_overlay.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=(160, 160, 160))
            
            # Apply severe Gaussian blur to make the lesion patch look like hazy fluid infiltration
            lesion_overlay = lesion_overlay.filter(ImageFilter.GaussianBlur(radius=15))
            
            # Blend lesion with base image
            img = Image.blend(img, Image.blend(img, lesion_overlay, 0.45), 0.5)
            
            # Draw the tight annotation mask (the ground-truth lesion core zone)
            draw_mask.ellipse([(cx - int(r*0.8), cy - int(r*0.8)), (cx + int(r*0.8), cy + int(r*0.8))], fill=255)
            
        # Draw thin rib cages lines to make it look realistic
        for y in range(120, 380, 45):
            draw.arc([(100, y - 40), (220, y + 40)], start=210, end=330, fill=(90, 90, 90), width=3)
            draw.arc([(292, y - 40), (412, y + 40)], start=210, end=330, fill=(90, 90, 90), width=3)
            
        # Post-process standard grayscale blurring to simulate radiograph noise
        xray_img = img.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5))
        xray_img = xray_img.convert("RGB") # Keep 3-channels for timm models
        
        return xray_img, mask, lesion_bbox

    def generate_patient_record(self, has_pneumonia: bool, noise_level: float = 0.2) -> Dict[str, Any]:
        """
        Generates a complete patient case.
        """
        # 1. Create radiographic scan and mask
        xray_img, mask, bbox = self.draw_mock_chest_xray(has_pneumonia)
        
        # Apply physical scanner degradation noise
        if noise_level > 0.05:
            arr = np.array(xray_img, dtype=np.float32)
            # Add Gaussian sensor noise
            noise = np.random.normal(0, 15 * noise_level, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            xray_img = Image.fromarray(arr)
            
        # 2. Generate structured vitals and lab markers
        age = random.randint(18, 85)
        
        if has_pneumonia:
            # Patients with pneumonia show high infection vitals
            temperature = round(random.uniform(100.5, 103.8), 1)
            heart_rate = random.randint(95, 125)
            wbc_count = random.randint(12500, 19500) # Elevated white blood cell count
            oxygen_sat = random.randint(86, 94)      # Reduced blood oxygenation
            
            # Clinical notes containing keywords
            transcripts = [
                f"Patient presents with severe dyspnea, fever of {temperature}F, and productive cough. Lung auscultation reveals crackles in lower lobe. Suspect severe acute infection.",
                f"Admitted with complaints of chest pain and high fever. Elevated heart rate observed. Radiograph indicates focal opacities in lungs. Diagnosis consistent with pneumonia.",
                f"Elderly female with altered mental status and low oxygen saturation ({oxygen_sat}%). Lab work confirms leukocytosis. Chest findings show lobar infiltration."
            ]
            clinical_note = random.choice(transcripts)
            label = 1
        else:
            # Healthy patient profile
            temperature = round(random.uniform(97.2, 99.0), 1)
            heart_rate = random.randint(62, 88)
            wbc_count = random.randint(4800, 9500)
            oxygen_sat = random.randint(96, 100)
            
            transcripts = [
                "Patient presents for routine physical checkup. Lungs are clear to auscultation bilaterally. No chest pain, cough, or dyspnea reported.",
                f"Outpatient review. Normal vital signs (Temp: {temperature}F). Normal chest expansion, clear breath sounds. No acute cardiopulmonary processes.",
                "Routine pre-operative clearance. Vital signs stable. Heart sounds normal, clear breath sounds. Chest X-ray reported negative for active disease."
            ]
            clinical_note = random.choice(transcripts)
            label = 0
            
        lab_metrics = {
            "age": float(age),
            "temperature": float(temperature),
            "heart_rate": float(heart_rate),
            "wbc_count": float(wbc_count),
            "oxygen_saturation": float(oxygen_sat)
        }
        
        # Clinical annotations metadata for self-healing verification
        metadata = [
            {"marker": "temperature", "value": temperature, "critical_threshold": 100.4, "operator": "greater"},
            {"marker": "oxygen_saturation", "value": oxygen_sat, "critical_threshold": 95, "operator": "less"},
            {"marker": "wbc_count", "value": wbc_count, "critical_threshold": 11000, "operator": "greater"},
            {"marker": "lesion_bbox", "value": bbox}
        ]
        
        return {
            "image": xray_img,
            "mask": mask,
            "clinical_note": clinical_note,
            "lab_metrics": lab_metrics,
            "metadata": metadata,
            "label": label
        }

if __name__ == "__main__":
    engine = PatientCraftEngine()
    sample = engine.generate_patient_record(has_pneumonia=True)
    print("PatientCraft Simulation Test:")
    print(f"Label: {sample['label']}")
    print(f"Lab Metrics: {sample['lab_metrics']}")
    print(f"Clinical Note: {sample['clinical_note']}")
    print(f"Mask values: {np.unique(np.array(sample['mask']))}")
