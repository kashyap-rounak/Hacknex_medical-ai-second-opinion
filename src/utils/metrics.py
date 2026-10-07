"""
Evaluation and Safety Metrics for Medical AI Grounding and Calibration.
Includes IoU, Pointing Game Accuracy, ECE, Brier Score, and Hallucination Suppression Rate.
"""

import numpy as np
from typing import List, Tuple, Dict

def calculate_iou(boxA: List[int], boxB: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)

    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def pointing_game_hit(heatmap: np.ndarray, ground_truth_box: List[int]) -> bool:
    """Check if the maximum activation point of the heatmap falls within the GT bounding box."""
    if heatmap is None or np.max(heatmap) == 0:
        return False
    # Find coordinate of maximum activation
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x1, y1, x2, y2 = ground_truth_box
    return bool(x1 <= x <= x2 and y1 <= y <= y2)

def calculate_ece(confidences: List[float], ground_truth: List[int], n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) across confidence bins."""
    if not confidences or len(confidences) != len(ground_truth):
        return 0.0

    confidences = np.array(confidences)
    ground_truth = np.array(ground_truth)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(ground_truth[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

def calculate_brier_score(confidences: List[float], ground_truth: List[int]) -> float:
    """Mean squared error between confidence predictions and binary outcomes."""
    if not confidences:
        return 0.0
    conf = np.array(confidences)
    gt = np.array(ground_truth)
    return float(np.mean((conf - gt) ** 2))

def evaluate_hallucination_safety(candidates: List[Dict]) -> Dict[str, float]:
    """
    Computes gate statistics:
    - total_candidates
    - supported_count
    - suppressed_count
    - suppression_rate
    - grounding_coverage
    """
    total = len(candidates)
    if total == 0:
        return {
            "total": 0,
            "supported": 0,
            "suppressed": 0,
            "suppression_rate": 0.0,
            "grounding_coverage": 0.0
        }
    
    supported = sum(1 for c in candidates if c.get("status") == "SUPPORTED")
    suppressed = total - supported
    grounded = sum(1 for c in candidates if c.get("visual_support") or c.get("text_support"))
    
    return {
        "total": total,
        "supported": supported,
        "suppressed": suppressed,
        "suppression_rate": suppressed / total,
        "grounding_coverage": grounded / total
    }
