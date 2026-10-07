"""
Medical Image Encoder Abstraction.
Uses Hugging Face CLIP (openai/clip-vit-base-patch32) as a lightweight production proxy.
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional
from app.config import settings
from PIL import Image

class MedicalImageEncoder:
    def __init__(self, device: Optional[str] = None):
        self.device = device or settings.resolved_device
        self.model_loaded = False
        self.processor = None
        self.model = None

    def _init_encoder(self):
        """Loads the CLIP vision model from Hugging Face."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPVisionModelWithProjection
            
            model_id = "openai/clip-vit-base-patch32"
            print(f"[*] Loading Production Vision Model: {model_id} on {self.device}")
            warnings.filterwarnings("ignore", category=UserWarning)
            
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(self.device)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"[!] Warning: Could not load CLIP Vision ({e}). Using feature fallback.")
            self.model_loaded = False

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        """
        Generates normalized image embedding vector.
        """
        if not self.model_loaded and self.processor is None:
            self._init_encoder()
            
        if self.model_loaded:
            import torch
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 4:
                image = image[..., :3]
                
            pil_img = Image.fromarray(image.astype('uint8'), 'RGB')
            with torch.no_grad():
                inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds
                image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            return image_embeds.cpu().numpy().flatten()
            
        # Resilient local fallback feature projection if download is pending
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        h, w, c = image.shape
        resized = np.array(image[::max(1, h//32), ::max(1, w//16), 0], dtype=np.float32)
        flat = resized.flatten()
        if len(flat) > 512:
            flat = flat[:512]
        elif len(flat) < 512:
            flat = np.pad(flat, (0, 512 - len(flat)))
        norm = flat / (np.linalg.norm(flat) + 1e-8)
        return norm

