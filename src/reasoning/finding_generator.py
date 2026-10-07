"""
Candidate Finding Generator.
Extracts potential findings across modalities before passing them into visual grounding.
"""

from typing import List, Dict, Any, Optional
from src.models.vlm import MultimodalVLM

class FindingGenerator:
    def __init__(self):
        self.vlm = MultimodalVLM()

    def generate_candidates(
        self,
        image_metadata: Dict[str, Any],
        clinical_notes: str,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Produce a list of raw candidate findings.
        Includes plausible findings and tests the pipeline's robustness with edge cases.
        """
        candidates = self.vlm.generate_candidate_findings(
            image_metadata=image_metadata,
            clinical_notes=clinical_notes,
            patient_metadata=patient_metadata
        )
        return candidates
