"""
Evidence Extractor.
Extracts grounded evidence spans from patient notes and spatial coordinates from the image.
"""

from typing import Dict, Any, List
from src.safety.evidence_validator import EvidenceValidator

class EvidenceExtractor:
    @staticmethod
    def extract_evidence(finding_name: str, clinical_notes: str, visual_grounding_res: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesizes visual and textual evidence into a single structured record."""
        text_eval = EvidenceValidator.validate_text_support(finding_name, clinical_notes)
        
        evidence_chain = {
            "finding": finding_name,
            "visual": {
                "grounding_score": visual_grounding_res.get("score", 0.0),
                "bbox": visual_grounding_res.get("bbox", [0, 0, 0, 0]),
                "explanation": visual_grounding_res.get("visual_explanation", "")
            },
            "clinical_notes": {
                "supported": text_eval["supported"],
                "score": text_eval["score"],
                "span": text_eval["span"],
                "source": text_eval["source"]
            }
        }
        return evidence_chain
