"""
Automated Unit Tests for Preprocessing Pipelines (2D X-ray, 3D CT, and Clinical Text).
"""

import numpy as np
import pytest
from src.preprocessing.image_preprocess import apply_clahe, standardize_image
from src.preprocessing.ct_preprocess import apply_hu_window, get_axial_slice, generate_synthetic_ct_volume
from src.preprocessing.text_preprocess import clean_clinical_text, extract_clinical_keywords

def test_clahe_enhancement():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    enhanced = apply_clahe(img)
    assert enhanced.shape == (100, 100, 3)
    assert enhanced.dtype == np.uint8

def test_standardize_image():
    img = np.full((200, 300, 3), 200, dtype=np.uint8)
    std = standardize_image(img, target_size=(256, 256))
    assert std.shape == (256, 256, 3)
    assert 0.0 <= std.min() and std.max() <= 1.0

def test_ct_volume_preprocessing():
    vol = generate_synthetic_ct_volume(num_slices=10, height=64, width=64)
    assert vol.shape == (10, 64, 64)
    
    axial_slice = get_axial_slice(vol, slice_index=5, window_type="lung")
    assert axial_slice.shape == (64, 64, 3)
    assert axial_slice.dtype == np.uint8

def test_clinical_text_cleaning():
    raw = "  Patient reports \n\n  cough   and dyspnea. \t\t "
    cleaned = clean_clinical_text(raw)
    assert cleaned == "Patient reports cough and dyspnea."

def test_clinical_keyword_extraction():
    text = "Patient has severe cardiomegaly and bilateral pleural effusion, but no fever."
    keywords = extract_clinical_keywords(text)
    assert "cardiomegaly" in keywords
    assert "pleural effusion" in keywords
