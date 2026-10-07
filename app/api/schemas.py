"""
FastAPI Request and Response Pydantic Schemas for MedSight AI.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class PatientMetadata(BaseModel):
    patient_id: str = Field(default="PT-10924", description="De-identified patient ID")
    age: Optional[int] = Field(default=68, description="Patient age")
    sex: Optional[str] = Field(default="M", description="Patient biological sex")

class AnalyzeRequest(BaseModel):
    clinical_notes: str = Field(
        default="Patient presents with worsening dyspnea, non-productive cough, and peripheral edema.",
        description="Clinical notes text"
    )
    patient_metadata: Optional[PatientMetadata] = None
    custom_candidates: Optional[List[str]] = None

class FindingResponse(BaseModel):
    finding: str
    confidence: float
    confidence_percentage: int
    confidence_level: str
    visual_support: bool
    visual_score: float
    text_support: bool
    text_score: float
    evidence_type: str
    bbox: List[int]
    segmentation_available: bool
    doctor_recommendation: str
    status: str

class AnalyzeResponse(BaseModel):
    case_id: str
    findings: List[FindingResponse]
    rejected_findings: List[Dict[str, Any]]
    total_candidates: int
    suppressed_count: int
    overall_summary: str
    safety_status: str
    hallucination_gate_active: bool

class ConfigResponse(BaseModel):
    mode: str
    model_device: str
    visual_threshold: float
    text_threshold: float
    confidence_high: float
    confidence_moderate: float
    enable_hallucination_gate: bool
    enable_ct: bool
