import numpy as np
import cv2

class GroundingEngine:
    def __init__(self, mode="demo"):
        self.mode = mode
        # In production, this would load BiomedCLIP / BioViL / MedSAM

    def ground(self, image_np: np.ndarray, finding_text: str) -> dict:
        '''
        Provides exact bounding boxes and visual scores.
        Fallback synthetic demo engine used when real models are not loaded.
        '''
        h, w = image_np.shape[:2]

        if "cardiomegaly" in finding_text.lower():
            bbox = [int(w*0.4), int(h*0.5), int(w*0.8), int(h*0.85)]
            score = 0.91
        elif "effusion" in finding_text.lower():
            bbox = [int(w*0.1), int(h*0.7), int(w*0.3), int(h*0.95)]
            score = 0.85
        elif "hallucinated" in finding_text.lower():
            bbox = [0, 0, 0, 0]
            score = 0.15 # Low visual grounding score
        else:
            bbox = [int(w*0.2), int(h*0.2), int(w*0.5), int(h*0.5)]
            score = 0.65

        heatmap = np.zeros((h, w), dtype=np.float32)
        if score > 0.5:
            cv2.rectangle(heatmap, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 1.0, -1)
            heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

        return {
            "bbox": bbox,
            "mask": heatmap > 0.5,
            "heatmap": heatmap,
            "score": score
        }