"""
Home Page and Case Selection for MedSight AI.
Provides project overview, architecture diagram, and pre-loaded Hackathon demo cases.
"""

import streamlit as st
from app.ui.components import render_disclaimer_banner

DEMO_CASES = {
    "Case 1: Cardiomegaly & Pulmonary Congestion (Visual Grounding Focus)": {
        "patient_id": "PT-77102",
        "age": 72,
        "sex": "Female",
        "notes": (
            "Patient with long-standing hypertension presents with 3-day history of increasing "
            "exertional dyspnea, orthopnea, and bilateral lower extremity pitting edema. "
            "Auscultation reveals basilar crackles. Request PA chest radiograph to evaluate for "
            "cardiomegaly and signs of congestive heart failure."
        ),
        "modality": "Chest X-ray",
        "expected": "Cardiomegaly with strong visual localization and clinical note concurrence."
    },
    "Case 2: Pleural Effusion & Cough (Text & Visual Multi-Source Support)": {
        "patient_id": "PT-88341",
        "age": 59,
        "sex": "Male",
        "notes": (
            "Patient reports persistent productive cough, pleuritic left-sided chest pain, and low-grade fevers. "
            "Decreased breath sounds at the left lung base. Clinical concern for left pleural effusion or consolidation."
        ),
        "modality": "Chest X-ray",
        "expected": "Left pleural effusion supported by costophrenic angle blunting and clinical history."
    },
    "Case 3: Safety Gate Stress Test (Demonstrates Hallucination Suppression)": {
        "patient_id": "PT-99415",
        "age": 45,
        "sex": "Female",
        "notes": (
            "Routine pre-operative clearance for elective knee arthroplasty. No respiratory complaints. "
            "Denies shortness of breath, cough, fever, or chest pain. Normal cardiopulmonary exam."
        ),
        "modality": "Chest X-ray",
        "expected": "Emergency hallucinated candidates (e.g. pneumothorax/free air) are HARD REJECTED by the gate."
    }
}

def render_home():
    """Renders the Home landing view."""
    render_disclaimer_banner()

    st.markdown("""
    ### 🔬 MedSight AI — Multimodal Medical Second Opinion
    **Visually Grounded Decision Support for Radiologists & Clinicians**
    
    MedSight AI is designed to act as an objective, evidence-grounded **second opinion** for doctors.
    It solves two fundamental failure modes of generic medical LLMs:
    1. **Unverifiable Claims:** Generic models generate text without pointing to anatomical coordinates. MedSight AI generates exact bounding boxes, continuous salience heatmaps, and segmentation masks.
    2. **Hallucination:** MedSight AI enforces a **Hard Evidence Gate**. A candidate finding **MUST NOT** reach the final clinical report unless it has verified visual ROI evidence or explicit documentation in clinical notes.
    """)

    st.markdown("---")
    st.subheader("⚡ 3-Minute Hackathon Demo Workflow")
    st.markdown("""
    1. **Navigate to 'Analyze Case'** in the sidebar.
    2. **Select a Demo Case** from the dropdown (or upload any chest X-ray / DICOM).
    3. **Click 'Analyze Case'** to execute multimodal reasoning.
    4. **Inspect the Visual Evidence:** Toggle between **Original Image**, **Salience Heatmap**, **Bounding Box ROI**, and **Segmentation Overlay**.
    5. **Observe the Hallucination Gate:** Notice how unsupported candidate findings are actively detected, suppressed, and logged in the audit trail!
    6. **Review the Final Report:** Read the physician advisory report formatted in doctor-assistant persona.
    """)

    st.markdown("---")
    st.subheader("📋 Pre-Configured Demo Cases")
    cols = st.columns(3)
    for idx, (c_name, c_data) in enumerate(DEMO_CASES.items()):
        with cols[idx]:
            st.markdown(f"#### {c_name.split(':')[0]}")
            st.caption(f"**Modality:** {c_data['modality']} | **Patient ID:** `{c_data['patient_id']}`")
            st.markdown(f"**Clinical Focus:** {c_data['expected']}")
            st.info(f"**Notes Excerpt:** \"{c_data['notes'][:90]}...\"")
