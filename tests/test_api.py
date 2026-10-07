"""
Automated Integration Tests for FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.api.routes import app

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "app" in data

def test_api_config():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "visual_threshold" in data
    assert "text_threshold" in data
    assert data["enable_hallucination_gate"] is True

def test_api_analyze_synthetic():
    response = client.post(
        "/api/analyze",
        data={
            "patient_id": "TEST_CASE_1",
            "clinical_notes": "Patient presents with chest pain and severe cough."
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["case_id"] == "TEST_CASE_1"
    assert "findings" in data
    assert "rejected_findings" in data
    assert data["hallucination_gate_active"] is True
