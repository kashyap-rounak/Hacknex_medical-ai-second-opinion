"""
Clinical Audit Logging for MedSight AI
Ensures HIPAA compliance by de-identifying logs and recording safety gate decisions.
"""

import logging
import sys
import re
from typing import Any, Dict

logger = logging.getLogger("MedSightAI")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [MedSight-Audit] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def anonymize_text(text: str) -> str:
    """Mask patient identifiers such as names, SSNs, phone numbers, and dates."""
    # Mask dates
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE_REDACTED]", text)
    # Mask SSN / MRN patterns
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN_REDACTED]", text)
    text = re.sub(r"\b(MRN|ID|Patient\s*ID)[:\s]+[A-Za-z0-9_-]+\b", r"\1: [ID_REDACTED]", text, flags=re.IGNORECASE)
    return text

def log_safety_decision(finding: str, status: str, visual_score: float, text_score: float, reason: str = ""):
    """Audit log entry for Hallucination Gate decisions."""
    logger.info(
        f"GATE_DECISION: finding='{finding}' | status={status} | "
        f"visual_score={visual_score:.3f} | text_score={text_score:.3f} | reason='{reason}'"
    )

def log_case_analysis(case_id: str, supported_count: int, rejected_count: int):
    """Audit log summary for an analyzed case."""
    clean_id = anonymize_text(case_id)
    logger.info(
        f"CASE_PROCESSED: case_id={clean_id} | supported={supported_count} | rejected={rejected_count}"
    )
