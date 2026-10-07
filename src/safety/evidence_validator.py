"""
Clinical Text Evidence Validator.
Determines whether finding claims are supported by patient notes with exact text span matching.
"""

import re
from typing import Dict, Any, Optional

class EvidenceValidator:
    @staticmethod
    def validate_text_support(finding: str, clinical_notes: str) -> Dict[str, Any]:
        """
        Calculates textual support score and extracts exact evidence span from notes.
        Example:
            finding: "Possible cardiomegaly"
            notes: "Patient reports persistent cough and shortness of breath. History of hypertension."
        """
        if not clinical_notes or not finding:
            return {"supported": False, "score": 0.0, "span": "", "source": "none"}

        finding_clean = finding.lower().strip()
        notes_clean = clinical_notes.lower()

        # Direct lexical match
        if finding_clean in notes_clean:
            return {
                "supported": True,
                "score": 0.95,
                "span": finding,
                "source": "clinical_notes"
            }

        # Synonym / Semantic clinical concept mapping
        concept_maps = {
            "cardiomegaly": ["heart", "enlarged heart", "hypertension", "cardiac", "chf"],
            "pleural effusion": ["effusion", "fluid", "shortness of breath", "cough", "dyspnea"],
            "consolidation": ["fever", "cough", "infiltrate", "pneumonia", "productive cough"],
            "edema": ["orthopnea", "swelling", "edema", "fluid retention"],
            "pneumothorax": ["sudden chest pain", "trauma", "collapsed lung"],
            "atelectasis": ["shallow breathing", "post-operative", "mucus plug"]
        }

        matched_terms = []
        for key, terms in concept_maps.items():
            if key in finding_clean:
                for term in terms:
                    if term in notes_clean:
                        matched_terms.append(term)

        if matched_terms:
            # Score proportional to specificity
            score = 0.85 if len(matched_terms) > 1 else 0.72
            # Find snippet
            first_term = matched_terms[0]
            start_pos = notes_clean.find(first_term)
            snippet_start = max(0, start_pos - 30)
            snippet_end = min(len(clinical_notes), start_pos + len(first_term) + 30)
            span_snippet = clinical_notes[snippet_start:snippet_end].strip()

            return {
                "supported": True,
                "score": score,
                "span": f"...{span_snippet}...",
                "source": "clinical_notes"
            }

        return {
            "supported": False,
            "score": 0.20,
            "span": "",
            "source": "none"
        }
