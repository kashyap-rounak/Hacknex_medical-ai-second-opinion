"""
Heatmap Generation and Spatial Salience Processing.
Transforms similarity scores into continuous spatial density maps.
"""

from typing import List
import cv2
import numpy as np

def generate_gaussian_roi_heatmap(
    h: int,
    w: int,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    peak_val: float = 1.0
) -> np.ndarray:
    """Creates a 2D anatomical Gaussian salience distribution."""
    y, x = np.ogrid[:h, :w]
    exponent = -(((x - center_x) ** 2) / (2 * (radius_x ** 2) + 1e-6) +
                 ((y - center_y) ** 2) / (2 * (radius_y ** 2) + 1e-6))
    heatmap = peak_val * np.exp(exponent)
    return heatmap.astype(np.float32)

def threshold_heatmap_to_roi(heatmap: np.ndarray, threshold: float = 0.5) -> List[int]:
    """Extracts bounding box [x1, y1, x2, y2] from heatmap activations above threshold."""
    binary = (heatmap >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0, 0, 0, 0]
    
    # Take the largest activated contour
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return [x, y, x + w, y + h]
