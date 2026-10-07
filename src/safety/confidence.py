"""
Confidence Calibration and Multi-Source Evidence Fusion Engine.
Calculates calibrated confidence scores by fusing visual grounding, text evidence,
prior model probabilities, and cross-modal agreement.
"""

from typing import Dict, Any, Optional
import numpy as np
from app.config import settings

class ConfidenceEngine:
    def __init__(
        self,
        high_threshold: float = None,
        moderate_threshold: float = None,
        temperature: float = 1.0
    ):
        self.high_threshold = high_threshold if high_threshold is not None else settings.CONFIDENCE_HIGH
        self.moderate_threshold = moderate_threshold if moderate_threshold is not None else settings.CONFIDENCE_MODERATE
        self.temperature = temperature

    def calibrate_score(self, raw_score: float) -> float:
        """Applies temperature scaling to raw confidence."""
        if self.temperature <= 0 or self.temperature == 1.0:
            return raw_score
        # Logit scaling
        p = np.clip(raw_score, 1e-4, 1.0 - 1e-4)
        logit = np.log(p / (1.0 - p))
        scaled_logit = logit / self.temperature
        calibrated = 1.0 / (1.0 + np.exp(-scaled_logit))
        return float(calibrated)

    def calculate(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fuses multimodal evidence sources into a single calibrated confidence metric.
        Formula:
            conf = 0.45 * visual_score + 0.25 * text_score + 0.20 * model_score + 0.10 * modality_agreement
        """
        v_score = float(finding_data.get("visual_score", 0.0))
        t_score = float(finding_data.get("text_score", 0.0))
        m_score = float(finding_data.get("model_score", 0.70))

        # Agreement indicator: 1.0 if both modalities are elevated (>0.5), 0.5 if single, 0.0 if neither
        v_active = v_score >= 0.5
        t_active = t_score >= 0.5
        if v_active and t_active:
            modality_agreement = 1.0
        elif v_active or t_active:
            modality_agreement = 0.5
        else:
            modality_agreement = 0.1

        # Multi-factor fusion
        fused = (
            0.45 * v_score +
            0.25 * t_score +
            0.20 * m_score +
            0.10 * modality_agreement
        )
        normalized = float(np.clip(fused, 0.0, 1.0))
        calibrated = self.calibrate_score(normalized)

        if calibrated >= self.high_threshold:
            level = "HIGH"
        elif calibrated >= self.moderate_threshold:
            level = "MODERATE"
        else:
            level = "LOW"

        finding_data["confidence"] = calibrated
        finding_data["confidence_percentage"] = int(round(calibrated * 100))
        finding_data["confidence_level"] = level
        finding_data["modality_agreement"] = modality_agreement

        return finding_data