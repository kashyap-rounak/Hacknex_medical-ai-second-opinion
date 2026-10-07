"""
MedSight AI — Main Streamlit Application Entrypoint.
Multimodal Medical Image Intelligence with Visual Grounding and Hard Hallucination Gate.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
import os

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import settings
from app.ui.home import render_home, DEMO_CASES
from app.ui.analysis import render_analysis_view
from app.ui.report import render_report_view
from app.ui.safety_dashboard import render_safety_dashboard
from app.ui.evaluation_view import render_evaluation_view
from src.reasoning.multimodal_reasoner import MultimodalReasoner
from src.ingestion.image_loader import ImageLoader
from src.preprocessing.ct_preprocess import generate_synthetic_ct_volume

# Page Setup
st.set_page_config(
    page_title="MedSight AI — Medical Second Opinion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Reasoner (cached)
@st.cache_resource
def get_reasoner():
    return MultimodalReasoner()

reasoner = get_reasoner()

# Session State Initialization
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "current_image" not in st.session_state:
    st.session_state["current_image"] = None
if "is_ct" not in st.session_state:
    st.session_state["is_ct"] = False
if "ct_volume" not in st.session_state:
    st.session_state["ct_volume"] = None

# Sidebar
with st.sidebar:
    st.title("🩺 MedSight AI")
    st.caption("Visually Grounded Clinical Decision Support")
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation:",
        ["1. Home Overview", "2. Analyze Case", "3. Evidence Viewer", "4. Clinical Report", "5. Safety Dashboard", "6. Evaluation"],
        index=1 if st.session_state["analysis_result"] else 0
    )

    st.markdown("---")
    st.subheader("📋 Case Input & Selection")

    demo_choice = st.selectbox(
        "Load Pre-Configured Case:",
        ["None (Custom Upload)"] + list(DEMO_CASES.keys()),
        index=1 # Preselect Case 1 for instant demo readiness
    )

    # Patient info fields
    if demo_choice != "None (Custom Upload)":
        preset = DEMO_CASES[demo_choice]
        default_pid = preset["patient_id"]
        default_age = preset["age"]
        default_sex = preset["sex"]
        default_notes = preset["notes"]
        modality_hint = preset["modality"]
    else:
        default_pid = "PT-10924"
        default_age = 65
        default_sex = "M"
        default_notes = "Patient reports persistent cough and shortness of breath. History of hypertension."
        modality_hint = "Chest X-ray"

    patient_id = st.text_input("Patient ID:", value=default_pid)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age:", value=default_age, min_value=1, max_value=120)
    with c2:
        sex = st.selectbox("Sex:", ["M", "F", "Other"], index=0 if default_sex == "M" else 1)

    clinical_notes = st.text_area("Clinical Notes & History:", value=default_notes, height=120)

    st.markdown("---")
    modality_type = st.radio("Modality:", ["2D Chest Radiograph (X-Ray)", "3D Computed Tomography (CT)"])
    
    uploaded_file = st.file_uploader(
        "Upload Medical Image (PNG, JPG, DICOM):",
        type=["png", "jpg", "jpeg", "dcm", "dicom"]
    )

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_clicked = st.button("🔍 Analyze Case", type="primary", use_container_width=True)
    with col_btn2:
        clear_clicked = st.button("🧹 Clear", use_container_width=True)

    if clear_clicked:
        st.session_state["analysis_result"] = None
        st.session_state["current_image"] = None
        st.session_state["is_ct"] = False
        st.session_state["ct_volume"] = None
        st.rerun()

    # System Status Footer
    st.markdown("---")
    st.caption(f"**Compute Device:** `{settings.resolved_device.upper()}`")
    st.caption(f"**Safety Gate:** `ACTIVE` (Visual: ≥ {settings.VISUAL_THRESHOLD}, Note: ≥ {settings.TEXT_THRESHOLD})")
    st.caption(f"**Mode:** `{settings.MODE.upper()}`")

# Logic: Perform Analysis
if analyze_clicked:
    with st.spinner("Executing Multimodal Evidence Grounding & Safety Gate..."):
        is_ct_mode = "3D" in modality_type
        st.session_state["is_ct"] = is_ct_mode

        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            img_np, _ = ImageLoader.load_from_bytes(bytes_data, uploaded_file.name)
            st.session_state["current_image"] = img_np
            st.session_state["ct_volume"] = None
        elif is_ct_mode:
            # Generate synthetic 3D CT volume
            ct_vol = generate_synthetic_ct_volume(num_slices=32, height=256, width=256)
            st.session_state["ct_volume"] = ct_vol
            # Select mid-slice as representative 2D projection
            img_np = (ct_vol[16, :, :]).copy()
            # Normalize for preview
            img_np = np.clip(img_np, -1000, 400)
            norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            img_np = (norm * 255).astype(np.uint8)
            img_np = np.stack([img_np] * 3, axis=-1)
            st.session_state["current_image"] = img_np
        else:
            # Default Anatomically Plausible Demo Chest X-ray
            # Synthetic realistic thoracic outline with rib cages and cardiac silhouette
            img_np = np.zeros((512, 512, 3), dtype=np.uint8)
            # Lung fields (darker)
            cv2.ellipse(img_np, (180, 240), (70, 150), 0, 0, 360, (50, 50, 50), -1)
            cv2.ellipse(img_np, (332, 240), (70, 150), 0, 0, 360, (50, 50, 50), -1)
            # Cardiac silhouette (central lower)
            cv2.ellipse(img_np, (270, 310), (85, 70), 20, 0, 360, (130, 130, 130), -1)
            # Rib lines
            for ry in range(160, 420, 40):
                cv2.ellipse(img_np, (256, ry), (160, 25), 0, 20, 160, (160, 160, 160), 2)
            # Spine & clavicles
            cv2.line(img_np, (256, 80), (256, 460), (180, 180, 180), 8)
            cv2.line(img_np, (120, 130), (392, 130), (170, 170, 170), 4)
            st.session_state["current_image"] = img_np
            st.session_state["ct_volume"] = None

        # Execute Reasoner
        patient_meta = {"patient_id": patient_id, "age": age, "sex": sex}
        result = reasoner.analyze_case(
            image=st.session_state["current_image"],
            clinical_notes=clinical_notes,
            patient_metadata=patient_meta
        )
        st.session_state["analysis_result"] = result
        st.success("Analysis complete! Review the visual evidence and report below.")

# Page Routing
if nav_choice == "1. Home Overview":
    render_home()

elif nav_choice in ["2. Analyze Case", "3. Evidence Viewer"]:
    if st.session_state["analysis_result"] is not None:
        render_analysis_view(
            analysis_result=st.session_state["analysis_result"],
            base_image=st.session_state["current_image"],
            is_ct=st.session_state["is_ct"],
            ct_volume=st.session_state["ct_volume"]
        )
    else:
        render_home()
        st.info("👈 Select a demo case or upload an image in the sidebar and click **'Analyze Case'** to view analysis.")

elif nav_choice == "4. Clinical Report":
    render_report_view(st.session_state["analysis_result"])

elif nav_choice == "5. Safety Dashboard":
    render_safety_dashboard(st.session_state["analysis_result"])

elif nav_choice == "6. Evaluation":
    render_evaluation_view()