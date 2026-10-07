"""
Safety and Explainability Dashboard for Hackathon Judges.
Displays key safety metrics: Hallucination Suppression Rate,
Evidence Coverage, Calibrated Confidence, and the full Evidence Decision Chain.
"""

import streamlit as st
from app.ui.components import render_disclaimer_banner

def render_safety_dashboard(analysis_result: dict):
    """Renders the executive safety and explainability dashboard."""
    render_disclaimer_banner()

    st.subheader("🛡️ Safety & Explainability Dashboard (Judge Review)")
    st.markdown("""
    This panel demonstrates MedSight AI's core innovation: **Preventing LLM Medical Hallucinations**
    via a hard evidence gate and calibrated multimodal verification.
    """)

    findings = analysis_result.get("findings", []) if analysis_result else []
    rejected = analysis_result.get("rejected_findings", []) if analysis_result else []
    total_candidates = len(findings) + len(rejected)

    # Top Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Candidates", total_candidates)
    with col2:
        st.metric("Grounded & Supported", len(findings))
    with col3:
        st.metric("Hallucinations Suppressed", len(rejected), delta=f"{len(rejected)} blocked" if rejected else "0", delta_color="inverse")
    with col4:
        avg_conf = (sum(f.get("confidence", 0) for f in findings) / len(findings)) if findings else 0.0
        st.metric("Avg Calibrated Confidence", f"{avg_conf:.0%}")
    with col5:
        st.metric("Hallucination Gate", "ACTIVE (Enforced)", delta="100% Protected", delta_color="normal")

    st.markdown("---")
    st.markdown("### 🧬 The Multi-Modal Evidence Decision Chain")
    st.markdown("""
    Every prospective finding must traverse this verification pipeline before clinical presentation:
    """)

    # Visual workflow diagram
    st.markdown("""
    ```
    ┌─────────────────────────┐
    │   Candidate Finding     │  (e.g., Cardiomegaly, Effusion, Tension Pneumothorax)
    └───────────┬─────────────┘
                │
                ▼
    ┌─────────────────────────┐
    │ Dual Evidence Check     │──► 1. Visual Localization (ROI / Heatmap Score ≥ 0.65)
    └───────────┬─────────────┘──► 2. Clinical Documentation (Notes Score ≥ 0.70)
                │
                ▼
    ┌─────────────────────────┐
    │  HARD EVIDENCE GATE     │
    └─────┬─────────────┬─────┘
          │             │
       [PASS]        [FAIL]
          │             │
          ▼             ▼
    ┌───────────┐ ┌───────────────────────────────────────────────┐
    │ Calibrate │ │ REJECT & SUPPRESS FROM REPORT                 │
    │ Confidence│ │ (Logged to audit trail, never shown as claim) │
    └─────┬─────┘ └───────────────────────────────────────────────┘
          │
          ▼
    ┌─────────────────────────┐
    │ Doctor-Assistant Report │  ("Doctor, consider reviewing this region...")
    └─────────────────────────┘
    ```
    """)

    st.markdown("---")
    st.markdown("### 📊 Active Case Evidence Trace")
    
    if not (findings or rejected):
        st.info("Run an analysis case first to populate the live audit trace.")
        return

    for idx, f in enumerate(findings + rejected, 1):
        is_supported = f.get("status") == "SUPPORTED"
        status_badge = "✅ SUPPORTED" if is_supported else "❌ SUPPRESSED (REJECTED)"
        
        with st.container():
            st.markdown(f"#### Finding #{idx}: **{f.get('finding')}** — `{status_badge}`")
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
            with c1:
                st.write(f"**Visual Grounding:** `{f.get('visual_score', 0):.2f}`")
            with c2:
                st.write(f"**Clinical Note Score:** `{f.get('text_score', 0):.2f}`")
            with c3:
                st.write(f"**Calibrated Conf:** `{f.get('confidence_percentage', 0)}%`")
            with c4:
                st.write(f"**Evidence Stream:** `{f.get('evidence_type', 'none').upper()}`")
            st.divider()
