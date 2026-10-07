"""
Automated Unit Tests for Visual Grounding and Bounding Box Operations.
"""

import numpy as np
import pytest
from src.models.grounding import GroundingEngine
from src.grounding.bbox import validate_bbox, bbox_to_relative, relative_to_bbox
from src.utils.metrics import calculate_iou, pointing_game_hit

@pytest.fixture
def dummy_image():
    return np.zeros((512, 512, 3), dtype=np.uint8)

def test_grounding_cardiomegaly(dummy_image):
    engine = GroundingEngine(mode="demo")
    res = engine.ground(dummy_image, "Possible cardiomegaly")
    assert res["score"] >= 0.85
    assert len(res["bbox"]) == 4
    x1, y1, x2, y2 = res["bbox"]
    assert x2 > x1 and y2 > y1
    assert res["heatmap"].shape == (512, 512)
    assert res["mask"].dtype == bool

def test_grounding_unsupported_finding(dummy_image):
    engine = GroundingEngine(mode="demo")
    res = engine.ground(dummy_image, "Tension pneumothorax")
    # Low score for absent finding
    assert res["score"] < 0.65

def test_bbox_validation():
    clamped = validate_bbox([-10, -5, 600, 700], 512, 512)
    assert clamped == [0, 0, 512, 512]

def test_bbox_coordinate_transform():
    rel = bbox_to_relative([100, 200, 300, 400], 500, 500)
    assert rel == [0.2, 0.4, 0.6, 0.8]
    reconstructed = relative_to_bbox(rel, 500, 500)
    assert reconstructed == [100, 200, 300, 400]

def test_iou_calculation():
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50]
    assert calculate_iou(box1, box2) == 1.0

    box3 = [100, 100, 150, 150]
    assert calculate_iou(box1, box3) == 0.0

def test_pointing_game():
    heatmap = np.zeros((100, 100), dtype=np.float32)
    heatmap[30, 40] = 1.0 # Peak at y=30, x=40
    gt_box = [20, 20, 60, 60]
    assert pointing_game_hit(heatmap, gt_box) is True

    miss_box = [70, 70, 90, 90]
    assert pointing_game_hit(heatmap, miss_box) is False
