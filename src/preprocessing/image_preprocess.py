"""
2D Medical Image Preprocessing.
Implements Contrast Limited Adaptive Histogram Equalization (CLAHE),
standardization, and tensor reshaping.
"""

import cv2
import numpy as np

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Enhance chest X-ray contrast using CLAHE on luminance channel."""
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    # RGB image: convert to LAB, apply to L channel, convert back
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def standardize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """Resize with bilinear interpolation and normalize values to [0.0, 1.0]."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    return norm
