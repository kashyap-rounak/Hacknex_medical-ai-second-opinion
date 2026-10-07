"""
Segmentation Mask Utilities.
Converts masks to contours, polygons, and computes coverage statistics.
"""

import cv2
import numpy as np
from typing import List, Dict, Any

def mask_to_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Extract polygon contours from binary mask."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def compute_mask_coverage(mask: np.ndarray) -> float:
    """Calculates the proportion of the image occupied by the mask."""
    if mask is None or mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)
