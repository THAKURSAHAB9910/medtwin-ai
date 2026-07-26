import os
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, List

class VLMExplainer:
    """
    Handles explainable AI post-processing for Medical VLMs.
    Draws visual grounding bounding boxes (from Florence-2) and formats
    conversational reasoning logs (from Qwen2-VL).
    """
    def __init__(self, output_dir: str = "docs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def draw_grounded_boxes(
        self, 
        image: Image.Image, 
        bboxes: List[List[int]], 
        label: str = "Pathology Lesion", 
        save_name: str = "vlm_grounded_output.png"
    ) -> str:
        """
        Overlays bounding boxes onto the radiographic image and saves the output.
        bboxes contains [xmin, ymin, xmax, ymax] formats.
        """
        # Create a copy to avoid mutating original image
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img)
        
        for box in bboxes:
            if len(box) == 4:
                xmin, ymin, xmax, ymax = box
                # Draw red bounding box rectangle
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                # Label overlay
                draw.text((xmin + 5, ymin + 5), label, fill="red")
                
        save_path = os.path.join(self.output_dir, save_name)
        annotated_img.save(save_path)
        return save_path

    def estimate_vlm_confidence(self, model_outputs: Dict[str, Any]) -> float:
        """
        Extracts prediction confidence or computes similarity probability estimates.
        """
        outputs = model_outputs.get("outputs", {})
        
        # If contrastive scores are present (BioMedCLIP), return maximum class probability
        if "contrastive_scores" in outputs:
            scores = list(outputs["contrastive_scores"].values())
            # Normalize scores via softmax
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)
            return float(np.max(probs))
            
        # Return direct confidence score if mapped
        return float(outputs.get("confidence", 0.75))

    def generate_clinical_trace(self, model_outputs: Dict[str, Any]) -> str:
        """
        Constructs a structured textual report detailing VLM reasoning and metadata.
        """
        model_name = model_outputs.get("model", "unknown")
        task = model_outputs.get("task", "unknown")
        outputs = model_outputs.get("outputs", {})
        
        trace = []
        trace.append("="*60)
        trace.append(f" CLINICAL VLM REASONING TRACE - {model_name.upper()} ")
        trace.append("="*60)
        trace.append(f" - Inference Task:  {task}")
        trace.append(f" - Target Diagnosis: {outputs.get('diagnosis', 'N/A')}")
        trace.append(f" - Model Confidence: {outputs.get('confidence', 0.0):.2%}")
        trace.append(f" - Reasoning Log:    {outputs.get('explanation', 'None')}")
        
        # Add layout details if LayoutLMv3
        if "extracted_entities" in outputs:
            trace.append(" - Document Layout Elements:")
            for k, v in outputs["extracted_entities"].items():
                trace.append(f"    * {k}: {v}")
                
        # Add grounding details if Florence-2
        if "grounded_box" in outputs:
            trace.append(f" - Grounded Coordinates: {outputs['grounded_box']}")
            
        trace.append("="*60)
        return "\n".join(trace)

# Simple helper inside file to bypass numpy requirement if run stand-alone
import numpy as np
