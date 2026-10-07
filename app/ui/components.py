"""
Reusable Streamlit UI Components for MedSight AI.
Follows clean, high-contrast, professional clinical dashboard styling.
"""

import streamlit as st
from typing import Dict, Any, List

def render_disclaimer_banner():
    """Renders prominent medical decision support disclaimer."""
    st.warning(
        "⚠️ **CLINICAL DECISION SUPPORT NOTICE:** AI-assisted second opinion only. "
        "Not a definitive diagnosis or replacement for professional clinical judgment. "
        "All visual evidence and recommendations must be validated by a licensed physician.",
        icon="🩺"
    )

def render_finding_card(finding: Dict[str, Any], index: int):
    """Renders an individual clinical finding card with grounding and confidence indicators."""
    f_name = finding.get("finding", "Unknown")
    conf_pct = finding.get("confidence_percentage", int(finding.get("confidence", 0) * 100))
    conf_level = finding.get("confidence_level", "MODERATE")
    evidence_type = finding.get("evidence_type", "visual").replace("_", " ").title()
    v_score = finding.get("visual_score", 0.0)
    t_score = finding.get("text_score", 0.0)
    bbox = finding.get("bbox", [0, 0, 0, 0])
    has_seg = finding.get("segmentation_available", False)
    doctor_rec = finding.get("doctor_recommendation", "")

    # Badge color based on confidence level
    badge_color = "#2e7d32" if conf_level == "HIGH" else "#f57c00" if conf_level == "MODERATE" else "#c62828"

    with st.expander(f"**{index}. {f_name}** — Confidence: {conf_pct}% ({conf_level})", expanded=True):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric("Calibrated Confidence", f"{conf_pct}%", delta=conf_level, delta_color="normal")
        with col2:
            st.metric("Visual Grounding Score", f"{v_score:.2f}", delta="Strong" if v_score > 0.75 else "Moderate")
        with col3:
            st.metric("Clinical Note Score", f"{t_score:.2f}", delta="Explicit" if t_score > 0.75 else "None")

        st.markdown(f"**Evidence Source:** `{evidence_type}` | **ROI Coordinates [x1,y1,x2,y2]:** `{bbox}` | **Segmentation:** `{'Refined' if has_seg else 'Bounding Box'}`")
        
        if finding.get("text_span"):
            st.info(f"**Clinical Note Evidence Span:** \"{finding['text_span']}\"")
            
        st.success(f"👨‍⚕️ **Physician Advisory:** {doctor_rec}")

def render_safety_gate_banner(rejected_findings: List[Dict[str, Any]], supported_count: int):
    """Renders the Hallucination Gate status banner showing active hallucination suppression."""
    if rejected_findings:
        st.error(
            f"🛡️ **HALLUCINATION GATE ACTIVE:** {len(rejected_findings)} unsupported candidate finding(s) "
            f"were **REJECTED and SUPPRESSED** because they lacked sufficient visual or note evidence.",
            icon="🛡️"
        )
        with st.expander("🔍 Inspect Suppressed Candidates (Audit Log)"):
            for rf in rejected_findings:
                st.markdown(
                    f"- ❌ **{rf['finding']}** — Visual Score: `{rf['visual_score']:.2f}` (req: ≥ 0.65) | "
                    f"Note Score: `{rf['text_score']:.2f}` (req: ≥ 0.70) | *Status: DROPPED*"
                )
    else:
        st.success(
            f"🛡️ **HALLUCINATION GATE ACTIVE:** All {supported_count} reported findings passed "
            f"the evidence verification gate.",
            icon="✅"
        )
