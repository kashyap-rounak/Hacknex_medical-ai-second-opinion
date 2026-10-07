"""
Hard Hallucination Safety Gate.
Enforces the mandatory evidence verification rule:
Every candidate finding MUST have verified visual evidence OR explicit textual evidence in clinical notes.
Unsupported candidate findings are HARD REJECTED and suppressed from the final report.
"""

from typing import Dict, Any, List
from app.config import settings
from src.utils.logging import log_safety_decision

class HallucinationGate:
    def __init__(
        self,
        visual_threshold: float = None,
        text_threshold: float = None,
        enabled: bool = None
    ):
        self.visual_threshold = visual_threshold if visual_threshold is not None else settings.VISUAL_THRESHOLD
        self.text_threshold = text_threshold if text_threshold is not None else settings.TEXT_THRESHOLD
        self.enabled = enabled if enabled is not None else settings.ENABLE_HALLUCINATION_GATE
        self.audit_log: List[Dict[str, Any]] = []

    def validate_finding(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validates finding using the instance's configured thresholds."""
        return self._evaluate(
            finding_data=finding_data,
            v_thresh=self.visual_threshold,
            t_thresh=self.text_threshold,
            enabled=self.enabled,
            audit_log=self.audit_log
        )

    @classmethod
    def validate(cls, finding_data: Dict[str, Any], visual_threshold: float = None, text_threshold: float = None) -> Dict[str, Any]:
        """Classmethod helper for static evaluations."""
        v_thresh = visual_threshold if visual_threshold is not None else settings.VISUAL_THRESHOLD
        t_thresh = text_threshold if text_threshold is not None else settings.TEXT_THRESHOLD
        return cls._evaluate(
            finding_data=finding_data,
            v_thresh=v_thresh,
            t_thresh=t_thresh,
            enabled=settings.ENABLE_HALLUCINATION_GATE,
            audit_log=None
        )

    @staticmethod
    def _evaluate(
        finding_data: Dict[str, Any],
        v_thresh: float,
        t_thresh: float,
        enabled: bool,
        audit_log: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        finding_name = finding_data.get("finding", "Unknown finding")
        v_score = float(finding_data.get("visual_score", 0.0))
        t_score = float(finding_data.get("text_score", 0.0))

        has_visual = v_score >= v_thresh
        has_text = t_score >= t_thresh

        finding_data["visual_support"] = has_visual
        finding_data["text_support"] = has_text

        if not enabled:
            finding_data["status"] = "SUPPORTED"
            finding_data["rejection_reason"] = "Gate disabled"
            return finding_data

        if has_visual or has_text:
            finding_data["status"] = "SUPPORTED"
            if has_visual and has_text:
                finding_data["evidence_type"] = "visual_and_clinical"
                finding_data["evidence_summary"] = "Concordant visual ROI and clinical note confirmation."
            elif has_visual:
                finding_data["evidence_type"] = "visual"
                finding_data["evidence_summary"] = f"Visually grounded with spatial localization score {v_score:.2f}."
            else:
                finding_data["evidence_type"] = "clinical_notes"
                finding_data["evidence_summary"] = f"Documented explicitly in clinical notes (score {t_score:.2f})."
            
            log_safety_decision(finding_name, "SUPPORTED", v_score, t_score, finding_data["evidence_summary"])
        else:
            finding_data["status"] = "REJECTED"
            finding_data["evidence_type"] = "none"
            finding_data["evidence_summary"] = (
                f"Suppressed: Insufficient evidence (visual: {v_score:.2f} < {v_thresh}, "
                f"text: {t_score:.2f} < {t_thresh})."
            )
            log_safety_decision(finding_name, "REJECTED", v_score, t_score, finding_data["evidence_summary"])

        if audit_log is not None:
            audit_log.append({
                "finding": finding_name,
                "status": finding_data["status"],
                "visual_score": v_score,
                "text_score": t_score,
                "visual_threshold": v_thresh,
                "text_threshold": t_thresh
            })

        return finding_data

    def filter_candidates(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes a list of candidate findings, partitioning into supported and suppressed collections."""
        supported = []
        rejected = []

        for c in candidates:
            validated = self.validate_finding(c)
            if validated["status"] == "SUPPORTED":
                supported.append(validated)
            else:
                rejected.append(validated)

        return {
            "supported_findings": supported,
            "rejected_findings": rejected,
            "total_candidates": len(candidates),
            "suppressed_count": len(rejected),
            "gate_passed": len(supported) > 0
        }