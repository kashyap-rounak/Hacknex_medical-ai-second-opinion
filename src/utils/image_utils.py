"""
Medical Image Processing and Visualization Utilities
Handles 2D bounding box drawing, heatmap blending, and segmentation mask overlays.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

def load_image_rgb(file_or_path) -> np.ndarray:
    """Load an image as an RGB NumPy array."""
    if isinstance(file_or_path, str):
        img = Image.open(file_or_path).convert("RGB")
    elif hasattr(file_or_path, "read"):
        img = Image.open(file_or_path).convert("RGB")
    elif isinstance(file_or_path, np.ndarray):
        if file_or_path.ndim == 2:
            return cv2.cvtColor(file_or_path, cv2.COLOR_GRAY2RGB)
        return file_or_path
    else:
        raise ValueError("Unsupported image source")
    return np.array(img)

def overlay_bounding_boxes(
    image: np.ndarray,
    boxes: List[List[int]],
    labels: List[str],
    scores: Optional[List[float]] = None,
    color: Tuple[int, int, int] = (0, 230, 118) # Clinical vibrant green
) -> np.ndarray:
    """Draw bounding boxes with labels and grounding scores on the image."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    for idx, bbox in enumerate(boxes):
        if not bbox or len(bbox) != 4 or bbox == [0, 0, 0, 0]:
            continue
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
        x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label badge
        score_text = f" ({scores[idx]:.0%})" if scores and idx < len(scores) else ""
        text = f"{labels[idx]}{score_text}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Background badge for legibility
        badge_y1 = max(0, y1 - th - baseline - 4)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 6)
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(annotated, text, (x1 + 3, y1 - baseline - 2), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return annotated

def blend_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """Blend a continuous 2D heatmap [0, 1] onto an RGB medical image."""
    h, w = image.shape[:2]
    # Resize heatmap if dimensions differ
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        
    norm_heat = np.clip(heatmap, 0.0, 1.0)
    heat_uint8 = (norm_heat * 255).astype(np.uint8)
    colored_heat = cv2.applyColorMap(heat_uint8, colormap)
    colored_heat = cv2.cvtColor(colored_heat, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(image, 1.0 - alpha, colored_heat, alpha, 0)
    return blended

def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 64, 129), # Cyan/Magenta highlight
    alpha: float = 0.4
) -> np.ndarray:
    """Overlay a binary segmentation mask with translucent tint and solid contour."""
    h, w = image.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    overlay = image.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    
    # Add contour border
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay.astype(np.uint8)
