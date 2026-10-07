"""
FastAPI Routes for MedSight AI Backend.
Provides /api/health, /api/config, and /api/analyze endpoints.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional, List
import json
import numpy as np
import cv2

from app.config import settings
from app.api.schemas import AnalyzeResponse, ConfigResponse, PatientMetadata
from src.ingestion.image_loader import ImageLoader
from src.reasoning.multimodal_reasoner import MultimodalReasoner

app = FastAPI(
    title="MedSight AI API",
    description="Multimodal Medical Image Intelligence with Hard Hallucination Gate",
    version=settings.APP_VERSION
)

reasoner = MultimodalReasoner()

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": settings.resolved_device,
        "mode": settings.MODE
    }

@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    return ConfigResponse(
        mode=settings.MODE,
        model_device=settings.resolved_device,
        visual_threshold=settings.VISUAL_THRESHOLD,
        text_threshold=settings.TEXT_THRESHOLD,
        confidence_high=settings.CONFIDENCE_HIGH,
        confidence_moderate=settings.CONFIDENCE_MODERATE,
        enable_hallucination_gate=settings.ENABLE_HALLUCINATION_GATE,
        enable_ct=settings.ENABLE_CT
    )

@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_case_api(
    clinical_notes: str = Form("Patient reports persistent cough and shortness of breath."),
    patient_id: str = Form("PT-10924"),
    patient_age: Optional[int] = Form(65),
    patient_sex: Optional[str] = Form("M"),
    file: Optional[UploadFile] = File(None)
):
    try:
        # Load image or generate synthetic demo image if not provided
        if file:
            contents = await file.read()
            img_np, _ = ImageLoader.load_from_bytes(contents, file.filename)
        else:
            # Synthetic 512x512 chest X-ray representation
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)
            cv2.putText(img_np, "DEMO CHEST X-RAY", (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        patient_meta = {
            "patient_id": patient_id,
            "age": patient_age,
            "sex": patient_sex
        }

        result = reasoner.analyze_case(
            image=img_np,
            clinical_notes=clinical_notes,
            patient_metadata=patient_meta
        )

        # Sanitize numpy arrays from JSON output
        clean_findings = [
            {k: v for k, v in f.items() if not isinstance(v, np.ndarray)}
            for f in result["findings"]
        ]
        clean_rejected = [
            {k: v for k, v in rf.items() if not isinstance(v, np.ndarray)}
            for rf in result["rejected_findings"]
        ]

        return AnalyzeResponse(
            case_id=result["case_id"],
            findings=clean_findings,
            rejected_findings=clean_rejected,
            total_candidates=result["total_candidates"],
            suppressed_count=result["suppressed_count"],
            overall_summary=result["overall_summary"],
            safety_status=result["safety_status"],
            hallucination_gate_active=result["hallucination_gate_active"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))