"""
Automated Unit Tests for Confidence Calibration Engine.
"""

import pytest
from src.safety.confidence import ConfidenceEngine

@pytest.fixture
def engine():
    return ConfidenceEngine(high_threshold=0.80, moderate_threshold=0.60)

def test_high_confidence_classification(engine):
    finding = {
        "visual_score": 0.95,
        "text_score": 0.90,
        "model_score": 0.85
    }
    result = engine.calculate(finding)
    assert result["confidence"] >= 0.80
    assert result["confidence_level"] == "HIGH"
    assert result["confidence_percentage"] >= 80

def test_moderate_confidence_classification(engine):
    finding = {
        "visual_score": 0.65,
        "text_score": 0.30,
        "model_score": 0.60
    }
    result = engine.calculate(finding)
    assert 0.50 <= result["confidence"] < 0.80
    assert result["confidence_level"] in ["MODERATE", "LOW"]

def test_low_confidence_classification(engine):
    finding = {
        "visual_score": 0.20,
        "text_score": 0.10,
        "model_score": 0.30
    }
    result = engine.calculate(finding)
    assert result["confidence"] < 0.60
    assert result["confidence_level"] == "LOW"

def test_temperature_scaling(engine):
    raw = 0.80
    # Temperature > 1 softens confidence towards 0.5
    engine.temperature = 2.0
    calibrated = engine.calibrate_score(raw)
    assert calibrated < raw
