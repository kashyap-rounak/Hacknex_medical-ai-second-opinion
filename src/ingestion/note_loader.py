"""
Clinical Note Parser and Ingestion.
Parses free-text notes, structures findings, and anonymizes identifiers.
"""

import re
from typing import Dict, List, Optional, Any
from src.utils.logging import anonymize_text

class ClinicalNoteLoader:
    @staticmethod
    def parse_note(raw_note: str) -> Dict[str, Any]:
        """Parse clinical notes into structured sections with anonymization."""
        if not raw_note:
            return {"clean_text": "", "sections": {}, "symptoms": [], "history": []}

        # Anonymize PHI
        clean_text = anonymize_text(raw_note.strip())

        # Section extraction (Chief Complaint, History, Exam, Impression)
        sections = {}
        current_section = "general"
        sections[current_section] = []

        lines = clean_text.splitlines()
        section_pattern = re.compile(r"^([A-Z\s]{3,30}):\s*(.*)$")

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            match = section_pattern.match(line_str)
            if match:
                sec_name = match.group(1).strip().lower()
                sec_val = match.group(2).strip()
                current_section = sec_name
                sections[current_section] = [sec_val] if sec_val else []
            else:
                sections.setdefault(current_section, []).append(line_str)

        structured_sections = {k: " ".join(v) for k, v in sections.items() if v}

        return {
            "clean_text": clean_text,
            "sections": structured_sections,
            "char_count": len(clean_text),
            "line_count": len(lines)
        }
