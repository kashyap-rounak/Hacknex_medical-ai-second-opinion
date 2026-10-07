"""
Evidence Audit Report Generator.
Provides judges and clinical auditors with a trace of the multimodal evidence chain.
"""

from typing import Dict, Any, List

class EvidenceReport:
    @staticmethod
    def generate_evidence_chain_table(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts findings into an audit table showing the evidence chain."""
        table = []
        for f in findings:
            table.append({
                "Finding": f.get("finding"),
                "Status": f.get("status"),
                "Evidence Source": f.get("evidence_type", "none").upper(),
                "Visual Score": round(f.get("visual_score", 0.0), 3),
                "Text Score": round(f.get("text_score", 0.0), 3),
                "Calibrated Confidence": f"{f.get('confidence_percentage', 0)}% ({f.get('confidence_level')})",
                "ROI Box": str(f.get("bbox", [])),
                "Segmentation": "Available" if f.get("segmentation_available") else "None"
            })
        return table
