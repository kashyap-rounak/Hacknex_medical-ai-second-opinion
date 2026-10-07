"""
Medical Guardrails and Doctor-Assistant Persona Enforcement.
Ensures that all generated communications respect clinical boundaries,
never claim definitive diagnosis, and explicitly prompt the physician for verification.
"""

import re

PROHIBITED_PHRASES = [
    r"\byou definitely have\b",
    r"\bthis proves\b",
    r"\byou have cancer\b",
    r"\bdiagnosis confirmed\b",
    r"\bpatient has\b",
    r"\bwe diagnose\b",
    r"\b100% certainty\b",
    r"\bdefinitive proof\b"
]

class MedicalGuardrails:
    @staticmethod
    def format_doctor_recommendation(finding_name: str, confidence_level: str, evidence_type: str) -> str:
        """Constructs respectful, doctor-assistant advisory language."""
        f_lower = finding_name.lower().replace("possible ", "")

        if confidence_level == "HIGH":
            support_desc = "substantial visual and clinical support"
        elif confidence_level == "MODERATE":
            support_desc = "moderate visual support"
        else:
            support_desc = "equivocal or preliminary support"

        recommendation = (
            f"Doctor, the system identified a region that may be consistent with {f_lower}. "
            f"The available evidence provides {support_desc} for this finding. "
            f"Consider reviewing the localized region of interest in correlation with the patient's full clinical presentation."
        )
        return recommendation

    @staticmethod
    def audit_and_sanitize_text(text: str) -> str:
        """Replaces definitive diagnostic claims with second-opinion framing."""
        sanitized = text
        for pattern in PROHIBITED_PHRASES:
            sanitized = re.sub(
                pattern,
                "findings may suggest or warrant physician review for",
                sanitized,
                flags=re.IGNORECASE
            )
        return sanitized
