"""
Secure Medical Image Loader
Validates file sizes, allowed mime types, and prevents memory exhaustion.
"""

import os
import io
import numpy as np
from PIL import Image
from typing import Tuple, Optional

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}
MAX_IMAGE_SIZE_BYTES = 50 * 1024 * 1024 # 50 MB limit

class ImageLoader:
    @staticmethod
    def validate_file(file_bytes: bytes, filename: str) -> None:
        """Validate input file size and extension."""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format: {ext}. Allowed: {ALLOWED_EXTENSIONS}")
        if len(file_bytes) > MAX_IMAGE_SIZE_BYTES:
            raise ValueError(f"File size exceeds maximum allowed ({MAX_IMAGE_SIZE_BYTES / (1024*1024):.0f}MB)")

    @staticmethod
    def load_from_bytes(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, dict]:
        """Load image bytes into RGB numpy array with metadata."""
        ImageLoader.validate_file(file_bytes, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext in {".dcm", ".dicom"}:
            from src.ingestion.dicom_loader import DicomLoader
            return DicomLoader.load_from_bytes(file_bytes)
        
        # Standard raster image
        img = Image.open(io.BytesIO(file_bytes))
        img = img.convert("RGB")
        img_np = np.array(img)
        
        meta = {
            "filename": filename,
            "width": img_np.shape[1],
            "height": img_np.shape[0],
            "channels": img_np.shape[2],
            "dtype": str(img_np.dtype)
        }
        return img_np, meta
