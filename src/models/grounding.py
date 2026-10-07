"""
Visual Grounding Model Interface and Spatial Attention Localizer.
Supports both fast anatomical heuristics (Demo/Test mode) and real OwlViT Zero-Shot Object Detection (Production mode).
"""

import cv2
import numpy as np
import warnings
from typing import Dict, Any, List, Optional
from app.config import settings
from PIL import Image

class GroundingEngine:
    def __init__(self, mode: Optional[str] = None):
        self.mode = mode or settings.MODE
        self.model_loaded = False
        self.processor = None
        self.model = None
        # Models will be lazily loaded on first inference call instead of blocking page startup


    def _init_model(self):
        try:
            import torch
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            
            print("[*] Loading Production Grounding Model: google/owlvit-base-patch32")
            warnings.filterwarnings("ignore", category=UserWarning)
            
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"[!] Warning: Could not load OwlViT ({e}). Falling back to anatomical heuristics.")

    def _ground_heuristic(self, h: int, w: int, phrase_clean: str):
        """Fast anatomical heuristic grounding for demo mode & tests."""
        if "cardiomegaly" in phrase_clean or "heart" in phrase_clean:
            x1, y1 = int(w * 0.32), int(h * 0.46)
            x2, y2 = int(w * 0.74), int(h * 0.84)
            score = 0.91
            desc = "Enlarged cardiac silhouette occupying >50% of thoracic diameter."
        elif "effusion" in phrase_clean:
            x1, y1 = int(w * 0.08), int(h * 0.68)
            x2, y2 = int(w * 0.34), int(h * 0.94)
            score = 0.84
            desc = "Blunting and meniscus sign at the costophrenic angle."
        elif "consolidation" in phrase_clean or "pneumonia" in phrase_clean:
            x1, y1 = int(w * 0.58), int(h * 0.35)
            x2, y2 = int(w * 0.86), int(h * 0.65)
            score = 0.78
            desc = "Airspace opacification in right mid-zone."
        elif "pneumothorax" in phrase_clean or "free air" in phrase_clean:
            x1, y1, x2, y2 = 0, 0, 0, 0
            score = 0.18
            desc = "No pleural line separation detected."
        elif "opacity" in phrase_clean or "edema" in phrase_clean:
            x1, y1 = int(w * 0.25), int(h * 0.30)
            x2, y2 = int(w * 0.75), int(h * 0.60)
            score = 0.68
            desc = "Perihilar haziness with vascular prominence."
        else:
            x1, y1 = int(w * 0.3), int(h * 0.3)
            x2, y2 = int(w * 0.7), int(h * 0.7)
            score = 0.45
            desc = "Non-specific diffuse pattern."
        return x1, y1, x2, y2, score, desc

    def ground(self, image: np.ndarray, finding_phrase: str) -> Dict[str, Any]:
        """
        Calculates spatial similarity and visual grounding for a clinical finding.
        """
        h, w = image.shape[:2]
        phrase_clean = finding_phrase.lower().strip()

        # Lazy load model on demand so application boots instantly
        if self.mode == "production" and not self.model_loaded and self.processor is None:
            self._init_model()

        # If in production and OwlViT is loaded, run real deep learning zero-shot detection
        if self.mode == "production" and self.model_loaded:

            import torch
            if image.ndim == 2:
                image_rgb = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image_rgb = image[..., :3]
            else:
                image_rgb = image
                
            pil_img = Image.fromarray(image_rgb.astype('uint8'), 'RGB')
            texts = [[f"a photo of a {phrase_clean}"]]
            
            with torch.no_grad():
                inputs = self.processor(text=texts, images=pil_img, return_tensors="pt")
                outputs = self.model(**inputs)
                target_sizes = torch.tensor([pil_img.size[::-1]])
                results = self.processor.post_process_grounded_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.01)[0]
                
            if len(results["scores"]) > 0:
                best_idx = results["scores"].argmax().item()
                score = results["scores"][best_idx].item()
                score = float(np.clip(score * 5.0, 0.0, 0.95))
                box = results["boxes"][best_idx].tolist()
                x1, y1, x2, y2 = [int(v) for v in box]
                desc = f"Zero-Shot Grounding Engine located region for '{phrase_clean}'."
                exec_mode = "PRODUCTION (OwlViT Zero-Shot)"
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0
                score = 0.15
                desc = f"No visual evidence identified for '{phrase_clean}'."
                exec_mode = "PRODUCTION (No Detections)"
        else:
            x1, y1, x2, y2, score, desc = self._ground_heuristic(h, w, phrase_clean)
            exec_mode = "PRODUCTION (Anatomical Heuristic)" if self.mode == "production" else "DEMO MODE"

        # Generate spatial similarity heatmap
        heatmap = np.zeros((h, w), dtype=np.float32)
        if score >= 0.5 and (x2 > x1) and (y2 > y1):
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), 1.0, -1)
            sigma = max(15, int(min(h, w) * 0.08))
            if sigma % 2 == 0:
                sigma += 1
            heatmap = cv2.GaussianBlur(heatmap, (sigma, sigma), 0)
            if np.max(heatmap) > 0:
                heatmap = (heatmap / np.max(heatmap)) * score

        mask = heatmap > (score * 0.55) if score >= 0.5 else np.zeros((h, w), dtype=bool)

        return {
            "bbox": [x1, y1, x2, y2],
            "mask": mask,
            "heatmap": heatmap,
            "score": score,
            "visual_explanation": desc,
            "execution_mode": exec_mode
        }

