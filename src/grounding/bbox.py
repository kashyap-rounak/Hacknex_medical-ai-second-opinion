"""
Bounding Box Geometry and Localization Utilities.
Provides coordinate transformations, Non-Maximum Suppression (NMS), and box validation.
"""

from typing import List, Tuple
import numpy as np

def validate_bbox(bbox: List[int], img_w: int, img_h: int) -> List[int]:
    """Clips and validates bounding box within image boundaries [x1, y1, x2, y2]."""
    if not bbox or len(bbox) != 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(img_w, x1))
    y1 = max(0, min(img_h, y1))
    x2 = max(x1, min(img_w, x2))
    y2 = max(y1, min(img_h, y2))
    return [x1, y1, x2, y2]

def bbox_to_relative(bbox: List[int], img_w: int, img_h: int) -> List[float]:
    """Convert absolute pixels to normalized [0.0, 1.0] coordinates."""
    x1, y1, x2, y2 = bbox
    return [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

def relative_to_bbox(rel_box: List[float], img_w: int, img_h: int) -> List[int]:
    """Convert normalized [0.0, 1.0] coordinates to absolute integer pixels."""
    rx1, ry1, rx2, ry2 = rel_box
    return [int(rx1 * img_w), int(ry1 * img_h), int(rx2 * img_w), int(ry2 * img_h)]

def calculate_box_area(bbox: List[int]) -> int:
    """Returns area in square pixels."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)
