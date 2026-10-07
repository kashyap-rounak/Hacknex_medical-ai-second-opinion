"""
Clinical Report View for MedSight AI.
Displays the structured doctor second-opinion report and allows exporting.
"""

import streamlit as st
from src.reporting.report_generator import ReportGenerator
from src.reporting.evidence_report import EvidenceReport
from app.ui.components import render_disclaimer_banner

def render_report_view(analysis_result: dict):
    """Renders the formal clinical report and evidence trace."""
    render_disclaimer_banner()

    if not analysis_result:
        st.info("No active case analysis. Please analyze a case first.")
        return

    findings = analysis_result.get("findings", [])
    rejected = analysis_result.get("rejected_findings", [])
    case_id = analysis_result.get("case_id", "CASE_001")

    st.subheader(f"📄 Clinical Second-Opinion Report: {case_id}")

    tab_report, tab_evidence, tab_json = st.tabs(["Formal Report", "Evidence Trace Table", "Raw JSON"])

    with tab_report:
        report_text = ReportGenerator.generate_clinical_report(analysis_result)
        st.code(report_text, language="markdown")
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"MedSight_Report_{case_id}.txt",
            mime="text/plain"
        )

    with tab_evidence:
        st.markdown("#### 🔗 Multimodal Evidence Chain (Audit Trace)")
        all_candidates = findings + rejected
        trace_table = EvidenceReport.generate_evidence_chain_table(all_candidates)
        st.dataframe(trace_table, use_container_width=True)

    with tab_json:
        # Serializable copy
        serializable = {
            "case_id": case_id,
            "findings": [
                {k: v for k, v in f.items() if k not in ["heatmap", "mask"]}
                for f in findings
            ],
            "rejected_findings": [
                {k: v for k, v in rf.items() if k not in ["heatmap", "mask"]}
                for rf in rejected
            ],
            "safety_status": analysis_result.get("safety_status"),
            "hallucination_gate_active": analysis_result.get("hallucination_gate_active")
        }
        st.json(serializable)
