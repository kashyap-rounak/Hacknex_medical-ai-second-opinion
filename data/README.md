# Data Management & Clinical Datasets

This repository supports multimodal medical datasets while strictly respecting data-use agreements and patient privacy.

## Supported Datasets

### 1. MS-CXR (Multi-modal Spatial Chest X-ray)
- **Source:** PhysioNet
- **Access Requirements:** Credentialed PhysioNet user with CITI Human Subjects Research training.
- **Description:** Ground truth bounding boxes paired with radiologist finding sentences across 8 cardinal findings.
- **Setup:**
  ```bash
  mkdir -p data/raw/ms-cxr
  # Download from PhysioNet and place annotations.csv in data/raw/ms-cxr/
  ```

### 2. VinDr-CXR
- **Source:** PhysioNet / Vingroup Big Data Institute
- **Access Requirements:** PhysioNet registration and Data Use Agreement.
- **Description:** 18,000 chest radiographs annotated with 22 local findings and 6 global diagnoses with radiologist consensus boxes.
- **Setup:**
  ```bash
  mkdir -p data/raw/vindr-cxr
  ```

### 3. MIMIC-CXR
- **Source:** PhysioNet / MIT Lab for Computational Physiology
- **Access Requirements:** Credentialed PhysioNet researcher.
- **Description:** 377,110 DICOM chest radiographs with associated clinical reports.
- **Setup:**
  ```bash
  mkdir -p data/raw/mimic-cxr
  ```

### 4. CT-RATE
- **Source:** HuggingFace / Mendeley Data
- **Access Requirements:** Open-access.
- **Description:** 3D CT volumes paired with radiology impressions for 3D volumetric evaluation.

## Demo / Offline Mode
By default, `MODE=demo` is enabled in `.env`.
To generate local synthetic demonstrator data without downloading external gigabyte-scale datasets:
```bash
python scripts/prepare_data.py
```
This populates `data/sample/` with anatomically valid test radiographs and manifest files.
