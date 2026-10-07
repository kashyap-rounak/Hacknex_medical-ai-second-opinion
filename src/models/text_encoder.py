"""
Biomedical Text Encoder Abstraction.
Uses Hugging Face CLIP (openai/clip-vit-base-patch32) to compute semantic text embeddings.
"""

import numpy as np
import warnings
from typing import List, Optional
from app.config import settings

class MedicalTextEncoder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or settings.resolved_device
        self.model_loaded = False
        self.processor = None
        self.model = None
        self._init_encoder()

    def _init_encoder(self):
        """Loads the CLIP text model from Hugging Face."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPTextModelWithProjection
            
            model_id = "openai/clip-vit-base-patch32"
            warnings.filterwarnings("ignore", category=UserWarning)
            
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPTextModelWithProjection.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.model_loaded = True
        except ImportError:
            self.model_loaded = False

    def encode_text(self, text: str) -> np.ndarray:
        """
        Embeds clinical finding or sentence into the same 512-dim embedding space as the image.
        """
        if not self.model_loaded:
            raise RuntimeError("Production model failed to load. Are torch and transformers installed?")
            
        import torch
        
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            
        return text_embeds.cpu().numpy().flatten()

    def compute_similarity(self, text_emb: np.ndarray, image_emb: np.ndarray) -> float:
        """Computes cosine similarity between real text and image embeddings."""
        dot = np.dot(text_emb, image_emb)
        # CLIP embeddings dot product is typically scaled by temperature (e.g. 100.0) in training,
        # but pure cosine similarity ranges [-1, 1]. Map it to a realistic probability proxy [0, 1].
        # For CLIP, dot products of matching pairs usually range from 0.2 to 0.4.
        # We scale it so a dot of 0.25+ becomes high confidence for the demo.
        score = float(np.clip((dot + 0.1) * 2.0, 0.0, 1.0))
        return score
