"""
Clinical Second-Opinion Report Generator.
Generates structured Markdown and text reports tailored for physician review.
"""

from typing import Dict, Any, List
from datetime import datetime

class ReportGenerator:
    @staticmethod
    def generate_clinical_report(analysis_result: Dict[str, Any], modality: str = "Chest X-ray") -> str:
        """Constructs the formal doctor-facing second-opinion report."""
        case_id = analysis_result.get("case_id", "ANONYMOUS_CASE")
        supported = analysis_result.get("findings", [])
        rejected = analysis_result.get("rejected_findings", [])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

        lines = [
            "=" * 64,
            "       MEDICAL AI SECOND-OPINION REPORT",
            "=" * 64,
            f"Case ID:        {case_id}",
            f"Modality:       {modality}",
            f"Generated:      {timestamp}",
            f"Safety Gate:    ACTIVE (Evidence Verification Enforced)",
            "-" * 64,
            "",
            "SUPPORTED CLINICAL FINDINGS (GROUNDED IN EVIDENCE):"
        ]

        if not supported:
            lines.append("No findings met the required visual or textual evidence thresholds.")
        else:
            for idx, f in enumerate(supported, 1):
                conf_pct = f.get("confidence_percentage", int(f.get("confidence", 0) * 100))
                lines.extend([
                    f"",
                    f"{idx}. {f['finding'].upper()}",
                    f"   Confidence:       {conf_pct}% ({f.get('confidence_level', 'MODERATE')})",
                    f"   Evidence Channel: {f.get('evidence_type', 'visual').replace('_', ' ').title()}",
                    f"   Visual Grounding: {f.get('visual_score', 0.0):.2f}",
                    f"   Note Support:     {f.get('text_score', 0.0):.2f}",
                    f"   ROI Box [x1,y1,x2,y2]: {f.get('bbox', [0, 0, 0, 0])}",
                    f"",
                    f"   Physician Advisory:",
                    f"   \"{f.get('doctor_recommendation', '')}\""
                ])

        lines.extend([
            "",
            "-" * 64,
            "SAFETY & HALLUCINATION CHECK:",
            f"✓ All reported findings ({len(supported)}) have verified supporting evidence."
        ])

        if rejected:
            lines.extend([
                f"",
                f"SUPPRESSED CANDIDATES ({len(rejected)}):",
                f"The following candidate(s) were REJECTED by the safety gate due to insufficient evidence:"
            ])
            for rf in rejected:
                lines.append(
                    f" - {rf['finding']}: (Visual Score: {rf['visual_score']:.2f}, Note Score: {rf['text_score']:.2f}) -> DROPPED"
                )
        else:
            lines.append("✓ No candidate findings were rejected.")

        lines.extend([
            "",
            "-" * 64,
            "DISCLAIMER:",
            "This AI-generated analysis is intended solely as clinical decision support",
            "for licensed healthcare professionals. It does not constitute a definitive diagnosis",
            "or replacement for clinical judgment. All regions of interest must be validated",
            "by the reviewing physician.",
            "=" * 64
        ])

        return "\n".join(lines)
