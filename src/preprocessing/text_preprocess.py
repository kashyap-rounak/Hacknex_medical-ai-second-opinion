"""
Clinical Text Preprocessing and Medical Keyword Extraction.
Cleans raw clinical transcriptions, extracts findings, symptoms, and anatomical references.
"""

import re
from typing import List, Set

COMMON_MEDICAL_ENTITIES = {
    "cardiomegaly", "enlarged heart", "cardiac silhouette",
    "pleural effusion", "effusion", "fluid accumulation",
    "pneumothorax", "collapsed lung",
    "consolidation", "infiltrate", "pneumonia",
    "edema", "pulmonary edema", "vascular congestion",
    "atelectasis", "volume loss", "collapse",
    "opacity", "hazy opacity", "ground glass",
    "nodule", "mass", "lesion",
    "cough", "shortness of breath", "dyspnea", "chest pain", "fever",
    "hypertension", "orthopnea"
}

def clean_clinical_text(text: str) -> str:
    """Normalize whitespace and remove non-clinical noisy characters."""
    if not text:
        return ""
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_clinical_keywords(text: str) -> List[str]:
    """Find key medical findings and symptoms explicitly cited in clinical notes."""
    cleaned = clean_clinical_text(text).lower()
    matches = []
    for entity in COMMON_MEDICAL_ENTITIES:
        if entity in cleaned:
            matches.append(entity)
    return sorted(list(set(matches)))
