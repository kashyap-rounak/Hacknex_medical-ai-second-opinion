"""
Multimodal Clinical Reasoner and Orchestration Layer.
Integrates image encoder, clinical text extraction, visual grounding,
the Hallucination Gate, confidence calibration, and doctor-assistant phrasing.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from src.models.grounding import GroundingEngine
from src.models.segmentation import MedicalSegmentationEngine
from src.reasoning.finding_generator import FindingGenerator
from src.safety.evidence_validator import EvidenceValidator
from src.safety.hallucination_gate import HallucinationGate
from src.safety.confidence import ConfidenceEngine
from src.safety.medical_guardrails import MedicalGuardrails
from src.utils.logging import log_case_analysis

class MultimodalReasoner:
    def __init__(self):
        self.grounding_engine = GroundingEngine()
        self.segmentation_engine = MedicalSegmentationEngine()
        self.finding_generator = FindingGenerator()
        self.hallucination_gate = HallucinationGate()
        self.confidence_engine = ConfidenceEngine()

    def analyze_case(
        self,
        image: np.ndarray,
        clinical_notes: str,
        patient_metadata: Optional[Dict[str, Any]] = None,
        custom_candidates: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes the full clinical second-opinion pipeline:
        1. Generates candidate findings
        2. Computes spatial visual grounding (heatmaps, bboxes, masks)
        3. Validates text/clinical notes evidence
        4. Applies HARD Hallucination Gate (drops unsupported claims)
        5. Computes calibrated confidence scores
        6. Formulates doctor-assistant advisory text
        """
        patient_id = (patient_metadata or {}).get("patient_id", "ANONYMOUS_CASE")
        
        # 1. Determine candidate findings
        if custom_candidates:
            candidates_raw = [{"finding": c, "category": "custom"} for c in custom_candidates]
        else:
            candidates_raw = self.finding_generator.generate_candidates(
                image_metadata={"shape": image.shape},
                clinical_notes=clinical_notes,
                patient_metadata=patient_metadata
            )

        processed_candidates = []

        # 2 & 3. Ground each candidate visually and textually
        for cand in candidates_raw:
            f_name = cand["finding"]
            
            # Visual Grounding
            g_res = self.grounding_engine.ground(image, f_name)
            v_score = g_res["score"]
            bbox = g_res["bbox"]
            heatmap = g_res["heatmap"]
            v_expl = g_res["visual_explanation"]

            # Segmentation mask inside ROI if applicable
            mask = self.segmentation_engine.segment_roi(image, bbox)

            # Clinical Note Entailment
            t_res = EvidenceValidator.validate_text_support(f_name, clinical_notes)
            t_score = t_res["score"]

            finding_obj = {
                "finding": f_name,
                "category": cand.get("category", "general"),
                "visual_score": v_score,
                "text_score": t_score,
                "bbox": bbox,
                "heatmap": heatmap,
                "mask": mask,
                "segmentation_available": bool(np.count_nonzero(mask) > 0),
                "visual_explanation": v_expl,
                "text_span": t_res["span"],
                "text_source": t_res["source"],
                "model_score": 0.82 # Baseline model prior
            }

            # 4. Filter through the Hard Hallucination Gate
            validated_obj = self.hallucination_gate.validate_finding(finding_obj)

            # 5. Calibrate confidence
            calibrated_obj = self.confidence_engine.calculate(validated_obj)

            # 6. Apply doctor-assistant persona formatting
            if calibrated_obj["status"] == "SUPPORTED":
                doctor_rec = MedicalGuardrails.format_doctor_recommendation(
                    finding_name=f_name,
                    confidence_level=calibrated_obj["confidence_level"],
                    evidence_type=calibrated_obj["evidence_type"]
                )
            else:
                doctor_rec = (
                    f"Candidate finding suppressed by safety gate. "
                    f"Neither visual ROI nor clinical documentation met the minimum evidence threshold."
                )
            calibrated_obj["doctor_recommendation"] = doctor_rec

            processed_candidates.append(calibrated_obj)

        # Partition findings
        supported_findings = [c for c in processed_candidates if c["status"] == "SUPPORTED"]
        rejected_findings = [c for c in processed_candidates if c["status"] == "REJECTED"]

        log_case_analysis(patient_id, len(supported_findings), len(rejected_findings))

        return {
            "case_id": patient_id,
            "findings": supported_findings,
            "rejected_findings": rejected_findings,
            "total_candidates": len(processed_candidates),
            "suppressed_count": len(rejected_findings),
            "hallucination_gate_active": True,
            "safety_status": "PASSED" if len(rejected_findings) == 0 else "SUPPRESSED_UNSUPPORTED_FINDINGS",
            "overall_summary": (
                f"Second-opinion analysis complete. {len(supported_findings)} grounded finding(s) identified. "
                f"{len(rejected_findings)} unsupported candidate finding(s) suppressed by safety gate."
            )
        }