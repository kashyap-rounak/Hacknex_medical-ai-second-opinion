from src.safety.hallucination_gate import HallucinationGate

def test_visual_pass():
    gate = HallucinationGate()
    res = gate.validate_finding({"visual_score": 0.90, "text_score": 0.0})
    assert res["status"] == "SUPPORTED"

def test_text_pass():
    gate = HallucinationGate()
    res = gate.validate_finding({"visual_score": 0.10, "text_score": 0.85})
    assert res["status"] == "SUPPORTED"

def test_reject():
    gate = HallucinationGate()
    res = gate.validate_finding({"visual_score": 0.30, "text_score": 0.20})
    assert res["status"] == "REJECTED"

def test_no_evidence_reject():
    gate = HallucinationGate()
    res = gate.validate_finding({"visual_score": 0.0, "text_score": 0.0})
    assert res["status"] == "REJECTED"