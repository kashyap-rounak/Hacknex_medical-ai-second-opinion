"""
Region of Interest (ROI) Inspection and Cropping Utilities.
Enables doctor-guided focused review of local image patches.
"""

import cv2
import numpy as np
from typing import List, Optional

def crop_roi(image: np.ndarray, bbox: List[int], padding_percent: float = 0.1) -> np.ndarray:
    """Crops the region of interest with optional contextual padding."""
    h, w = image.shape[:2]
    if not bbox or bbox == [0, 0, 0, 0] or len(bbox) != 4:
        return image

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = int(box_w * padding_percent)
    pad_y = int(box_h * padding_percent)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    cropped = image[cy1:cy2, cx1:cx2]
    return cropped if cropped.size > 0 else image
