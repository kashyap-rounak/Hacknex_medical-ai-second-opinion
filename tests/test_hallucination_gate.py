"""
Automated Unit Tests for Hallucination Gate.
Validates the Hard Evidence Verification rule:
- Visual score >= 0.65 -> PASS
- Text score >= 0.70 -> PASS
- Visual 0.30 & Text 0.20 -> REJECT
- No evidence -> REJECT
"""

import pytest
from src.safety.hallucination_gate import HallucinationGate

@pytest.fixture
def gate():
    return HallucinationGate(visual_threshold=0.65, text_threshold=0.70, enabled=True)

def test_visual_evidence_passes(gate):
    finding = {
        "finding": "Possible cardiomegaly",
        "visual_score": 0.90,
        "text_score": 0.00
    }
    result = gate.validate_finding(finding)
    assert result["status"] == "SUPPORTED"
    assert result["visual_support"] is True
    assert result["text_support"] is False
    assert result["evidence_type"] == "visual"

def test_text_evidence_passes(gate):
    finding = {
        "finding": "Pleural effusion",
        "visual_score": 0.10,
        "text_score": 0.85
    }
    result = gate.validate_finding(finding)
    assert result["status"] == "SUPPORTED"
    assert result["visual_support"] is False
    assert result["text_support"] is True
    assert result["evidence_type"] == "clinical_notes"

def test_dual_evidence_passes(gate):
    finding = {
        "finding": "Consolidation",
        "visual_score": 0.80,
        "text_score": 0.75
    }
    result = gate.validate_finding(finding)
    assert result["status"] == "SUPPORTED"
    assert result["evidence_type"] == "visual_and_clinical"

def test_insufficient_evidence_rejected(gate):
    finding = {
        "finding": "Hallucinated subdiaphragmatic free air",
        "visual_score": 0.30,
        "text_score": 0.20
    }
    result = gate.validate_finding(finding)
    assert result["status"] == "REJECTED"
    assert result["visual_support"] is False
    assert result["text_support"] is False
    assert result["evidence_type"] == "none"

def test_zero_evidence_rejected(gate):
    finding = {
        "finding": "Random ungrounded claim",
        "visual_score": 0.0,
        "text_score": 0.0
    }
    result = gate.validate_finding(finding)
    assert result["status"] == "REJECTED"
    assert result["status"] != "SUPPORTED"

def test_filter_candidates_partition(gate):
    candidates = [
        {"finding": "Cardiomegaly", "visual_score": 0.92, "text_score": 0.10},
        {"finding": "Unsupported Mass", "visual_score": 0.25, "text_score": 0.15},
        {"finding": "Pleural Effusion", "visual_score": 0.15, "text_score": 0.88}
    ]
    partition = gate.filter_candidates(candidates)
    assert len(partition["supported_findings"]) == 2
    assert len(partition["rejected_findings"]) == 1
    assert partition["rejected_findings"][0]["finding"] == "Unsupported Mass"
