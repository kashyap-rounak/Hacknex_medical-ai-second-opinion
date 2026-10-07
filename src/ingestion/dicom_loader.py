"""
DICOM Ingestion and Metadata De-identification.
Handles pixel array decoding and strips Protected Health Information (PHI).
"""

import io
import numpy as np
from typing import Tuple, Dict, Any

class DicomLoader:
    @staticmethod
    def load_from_bytes(dicom_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Parses DICOM byte stream. If pydicom is installed, extracts real pixel data.
        Otherwise falls back cleanly to synthetic standard representation for demo resilience.
        """
        try:
            import pydicom
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            pixel_array = ds.pixel_array.astype(np.float32)

            # Rescale intercept & slope if present
            slope = getattr(ds, "RescaleSlope", 1)
            intercept = getattr(ds, "RescaleIntercept", 0)
            pixel_array = pixel_array * slope + intercept

            # Normalize to 0-255 uint8 for 2D visualization
            norm = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min() + 1e-8)
            img_np = (norm * 255).astype(np.uint8)
            if img_np.ndim == 2:
                img_np = np.stack([img_np]*3, axis=-1)

            # De-identified metadata
            meta = {
                "Modality": str(getattr(ds, "Modality", "CR")),
                "BodyPartExamined": str(getattr(ds, "BodyPartExamined", "CHEST")),
                "PatientAge": str(getattr(ds, "PatientAge", "[REDACTED]")),
                "PatientSex": str(getattr(ds, "PatientSex", "U")),
                "PatientID": "[ANONYMIZED_DICOM_ID]",
                "PatientName": "[ANONYMIZED]",
                "Rows": getattr(ds, "Rows", img_np.shape[0]),
                "Columns": getattr(ds, "Columns", img_np.shape[1]),
            }
            return img_np, meta

        except ImportError:
            # Clean fallback when pydicom isn't available
            fallback = np.zeros((512, 512, 3), dtype=np.uint8)
            meta = {
                "Modality": "DX",
                "BodyPartExamined": "CHEST",
                "PatientID": "[ANONYMIZED_DEMO_ID]",
                "Status": "PYDICOM_UNAVAILABLE_FALLBACK"
            }
            return fallback, meta
