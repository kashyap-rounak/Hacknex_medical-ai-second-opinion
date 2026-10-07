"""
3D Computed Tomography (CT) Preprocessing and Volume Windowing.
Provides Hounsfield Unit (HU) normalization, lung/mediastinum windowing,
and axial slice extraction.
"""

import numpy as np
from typing import Tuple, Optional

# Standard Medical CT Windows (Center, Width)
CT_WINDOWS = {
    "lung": (-600, 1500),         # Window Center: -600, Window Width: 1500
    "mediastinum": (50, 350),     # Window Center: 50, Window Width: 350
    "bone": (400, 1800),          # Window Center: 400, Window Width: 1800
    "soft_tissue": (40, 400)      # Window Center: 40, Window Width: 400
}

def apply_hu_window(volume: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Window CT volume Hounsfield Units into [0, 255] range."""
    val_min = window_center - (window_width / 2.0)
    val_max = window_center + (window_width / 2.0)
    windowed = np.clip(volume, val_min, val_max)
    normalized = (windowed - val_min) / (val_max - val_min + 1e-8)
    return (normalized * 255).astype(np.uint8)

def get_axial_slice(volume: np.ndarray, slice_index: int, window_type: str = "lung") -> np.ndarray:
    """Extract and window a 2D axial slice from a 3D CT volume (D, H, W)."""
    depth = volume.shape[0]
    slice_idx = max(0, min(depth - 1, slice_index))
    raw_slice = volume[slice_idx, :, :]
    
    center, width = CT_WINDOWS.get(window_type, (-600, 1500))
    windowed_slice = apply_hu_window(raw_slice, center, width)
    
    # Expand to 3-channel RGB for visualization
    if windowed_slice.ndim == 2:
        return np.stack([windowed_slice] * 3, axis=-1)
    return windowed_slice

def generate_synthetic_ct_volume(num_slices: int = 32, height: int = 256, width: int = 256) -> np.ndarray:
    """Generates an anatomically plausible synthetic 3D CT volume for demo/testing without large datasets."""
    volume = np.full((num_slices, height, width), -1000.0, dtype=np.float32) # Air background
    
    # Simulate chest wall and lung parenchyma
    cy, cx = height // 2, width // 2
    for z in range(num_slices):
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Body contour (soft tissue HU ~ +40)
        body_mask = dist_from_center < (min(height, width) * 0.42)
        volume[z, body_mask] = 40.0
        
        # Bilateral lung fields (HU ~ -700)
        left_lung = np.sqrt((x - (cx - 45))**2 + (y - cy)**2) < 40
        right_lung = np.sqrt((x - (cx + 45))**2 + (y - cy)**2) < 40
        volume[z, left_lung | right_lung] = -700.0
        
        # Mediastinum / Spine (HU ~ +300)
        spine = np.sqrt((x - cx)**2 + (y - (cy + 60))**2) < 15
        volume[z, spine] = 300.0
        
    return volume
