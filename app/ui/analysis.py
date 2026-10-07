"""
Analysis Screen for MedSight AI.
Supports 2D Chest Radiograph and 3D CT Volume inspection,
multi-layer grounding overlays (Heatmap, BBox, Segmentation), and ROI cropping.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from app.ui.components import render_finding_card, render_safety_gate_banner, render_disclaimer_banner
from src.utils.image_utils import (
    overlay_bounding_boxes,
    blend_heatmap_on_image,
    overlay_mask
)
from src.grounding.roi import crop_roi
from src.preprocessing.ct_preprocess import get_axial_slice, generate_synthetic_ct_volume

def render_analysis_view(analysis_result: dict, base_image: np.ndarray, is_ct: bool = False, ct_volume: np.ndarray = None):
    """Renders the comprehensive medical analysis view."""
    render_disclaimer_banner()

    findings = analysis_result.get("findings", [])
    rejected = analysis_result.get("rejected_findings", [])
    case_id = analysis_result.get("case_id", "CASE_001")

    st.subheader(f"🔍 Case Analysis: {case_id}")
    render_safety_gate_banner(rejected, len(findings))

    # Controls row
    col_img, col_findings = st.columns([1.3, 1.0])

    with col_img:
        st.markdown("#### 🖼️ Visual Grounding Inspector")
        
        # Mode selector tabs/radios
        overlay_mode = st.radio(
            "Visualization Layer:",
            ["Annotated Overlay", "Salience Heatmap", "Bounding Boxes", "Segmentation Mask", "Original Image"],
            horizontal=True
        )

        display_image = base_image.copy()

        # If 3D CT, show slice slider
        if is_ct and ct_volume is not None:
            max_slices = ct_volume.shape[0]
            slice_idx = st.slider("Axial CT Slice:", min_value=0, max_value=max_slices - 1, value=max_slices // 2)
            window_choice = st.selectbox("HU Window:", ["lung", "mediastinum", "bone", "soft_tissue"])
            display_image = get_axial_slice(ct_volume, slice_idx, window_choice)

        # Apply chosen visualization
        if overlay_mode == "Original Image":
            st.image(display_image, caption="Original Medical Image (Unannotated)", use_container_width=True)

        elif overlay_mode == "Salience Heatmap":
            # Combine heatmaps from supported findings
            h, w = display_image.shape[:2]
            combined_heat = np.zeros((h, w), dtype=np.float32)
            for f in findings:
                if f.get("heatmap") is not None:
                    combined_heat = np.maximum(combined_heat, f["heatmap"])
            blended = blend_heatmap_on_image(display_image, combined_heat, alpha=0.45)
            st.image(blended, caption="Spatial Attention & Likelihood Heatmap", use_container_width=True)

        elif overlay_mode == "Bounding Boxes":
            boxes = [f["bbox"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            labels = [f["finding"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            scores = [f["visual_score"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            boxed_img = overlay_bounding_boxes(display_image, boxes, labels, scores)
            st.image(boxed_img, caption="Grounded Bounding Box Regions of Interest", use_container_width=True)

        elif overlay_mode == "Segmentation Mask":
            h, w = display_image.shape[:2]
            combined_mask = np.zeros((h, w), dtype=bool)
            for f in findings:
                if f.get("mask") is not None:
                    combined_mask = combined_mask | f["mask"]
            mask_img = overlay_mask(display_image, combined_mask, color=(255, 64, 129), alpha=0.4)
            st.image(mask_img, caption="Segmentation Contours (MedSAM / Adaptive Threshold)", use_container_width=True)

        elif overlay_mode == "Annotated Overlay":
            # Combined: heatmap + box + labels
            h, w = display_image.shape[:2]
            combined_heat = np.zeros((h, w), dtype=np.float32)
            for f in findings:
                if f.get("heatmap") is not None:
                    combined_heat = np.maximum(combined_heat, f["heatmap"])
            temp_blend = blend_heatmap_on_image(display_image, combined_heat, alpha=0.35)
            boxes = [f["bbox"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            labels = [f["finding"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            scores = [f["visual_score"] for f in findings if f.get("bbox") and f["bbox"] != [0, 0, 0, 0]]
            full_overlay = overlay_bounding_boxes(temp_blend, boxes, labels, scores)
            st.image(full_overlay, caption="Complete Multimodal Grounding Overlay", use_container_width=True)

        # Doctor ROI Inspection Tool
        if findings:
            st.markdown("---")
            st.markdown("#### 🔬 Focused ROI Crop Inspection")
            selected_f = st.selectbox(
                "Select Finding to Zoom into ROI:",
                options=[f["finding"] for f in findings],
                index=0
            )
            matching_finding = next((f for f in findings if f["finding"] == selected_f), None)
            if matching_finding and matching_finding.get("bbox") and matching_finding["bbox"] != [0, 0, 0, 0]:
                cropped_patch = crop_roi(display_image, matching_finding["bbox"], padding_percent=0.15)
                st.image(
                    cropped_patch,
                    caption=f"Magnified ROI: {selected_f} (Box: {matching_finding['bbox']})",
                    width=350
                )
                st.caption(f"Visual Grounding Evidence: {matching_finding.get('visual_explanation', '')}")

    with col_findings:
        st.markdown(f"#### 📋 Supported Findings ({len(findings)})")
        if not findings:
            st.info("No findings reached the required confidence or evidence thresholds.")
        else:
            for idx, f in enumerate(findings, 1):
                render_finding_card(f, idx)
