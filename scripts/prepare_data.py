"""
Dataset Ingestion, Adapter, and Sample Preparation Script.
Guides users on credentialed access (PhysioNet CITI training) and formats data
for MS-CXR, VinDr-CXR, MIMIC-CXR, and CT-RATE.
"""

import os
import json
import numpy as np
import cv2

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

SUPPORTED_DATASETS = {
    "MS-CXR": {
        "access": "Credentialed (PhysioNet)",
        "url": "https://physionet.org/content/ms-cxr/0.1/",
        "annotations": "Sentence-level bounding box ground truth annotations for 8 findings."
    },
    "VinDr-CXR": {
        "access": "Open / Registration (PhysioNet)",
        "url": "https://physionet.org/content/vindr-cxr/1.0.0/",
        "annotations": "18,000 postero-anterior chest X-rays with radiologist bounding boxes."
    },
    "MIMIC-CXR": {
        "access": "Credentialed (CITI Data or Specimens Training)",
        "url": "https://physionet.org/content/mimic-cxr/2.0.0/",
        "annotations": "377,110 chest radiographs paired with free-text radiology reports."
    },
    "CT-RATE": {
        "access": "Open-access 3D CT",
        "url": "https://huggingface.co/datasets/ibrahimethemhamamci/CT-RATE",
        "annotations": "3D chest CT volumes with radiologist impressions."
    }
}

def generate_sample_dataset():
    """Generates synthetic demonstrator chest radiographs and structured cases."""
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("[*] Generating synthetic sample cases for immediate demonstration...")

    # Sample Case 1: Cardiomegaly
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.ellipse(img1, (180, 240), (70, 150), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img1, (332, 240), (70, 150), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img1, (270, 310), (95, 75), 20, 0, 360, (130, 130, 130), -1)
    cv2.imwrite(os.path.join(SAMPLE_DIR, "sample_cardiomegaly.png"), img1)

    # Metadata
    sample_manifest = [
        {
            "case_id": "CASE_SAMPLE_01",
            "image_file": "sample_cardiomegaly.png",
            "findings": ["Possible cardiomegaly"],
            "ground_truth_bbox": [163, 235, 378, 430],
            "clinical_notes": "Patient with hypertension and ankle edema. Evaluate cardiac size."
        }
    ]

    with open(os.path.join(SAMPLE_DIR, "manifest.json"), "w") as f:
        json.dump(sample_manifest, f, indent=2)

    print(f"[SUCCESS] Created sample data in {os.path.abspath(SAMPLE_DIR)}")

if __name__ == "__main__":
    generate_sample_dataset()
