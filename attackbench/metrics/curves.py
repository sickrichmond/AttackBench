"""
Robust accuracy curves and evaluation metrics.
Ported from analysis/plot_distances.py logic
"""

from typing import Dict, List, Optional

import numpy as np

# NumPy 2.0+ compatibility: trapz was renamed to trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def compute_robust_accuracy_curve(
    distances: np.ndarray, success_mask: np.ndarray, num_points: int = 100
) -> Dict[str, List[float]]:
    """
    Compute robust accuracy curve from attack distances.
    Ported from analysis/plot_distances.py
    """
    if len(distances) == 0:
        return {"thresholds": [], "robust_accuracies": []}

    # Only consider successful attacks for distance computation
    successful_distances = distances[success_mask]
    successful_distances = successful_distances[np.isfinite(successful_distances)]

    if len(successful_distances) == 0:
        # No successful attacks - 100% robust accuracy at all thresholds
        finite_distances = distances[np.isfinite(distances)]
        max_dist = float(np.max(finite_distances)) if len(finite_distances) else 1.0
        # A zero-width curve is not useful for integration or plotting. This also
        # covers the common all-failed representation: every distance is infinity.
        if max_dist <= 0:
            max_dist = 1.0
        thresholds = np.linspace(0, max_dist, num_points)
        return {
            "thresholds": thresholds.tolist(),
            "robust_accuracies": [1.0] * num_points,
        }

    # Use the logic from plot_distances.py
    distances_unique, counts = np.unique(successful_distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(distances)

    return {
        "thresholds": distances_unique.tolist(),
        "robust_accuracies": robust_acc.tolist(),
    }


def compute_auc_robust_accuracy(
    thresholds: List[float], robust_accuracies: List[float]
) -> float:
    """Compute Area Under Curve for robust accuracy using trapezoidal rule."""
    if len(thresholds) < 2 or len(robust_accuracies) < 2:
        return 0.0

    return float(_trapz(robust_accuracies, thresholds))


def compute_certified_robustness_metrics(
    distances: np.ndarray,
    success_mask: np.ndarray,
    threat_model: str,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute certified robustness at specific thresholds."""
    if thresholds is None:
        # Default thresholds based on threat model
        if threat_model == "linf":
            thresholds = [4 / 255, 8 / 255, 16 / 255]
        elif threat_model == "l2":
            thresholds = [0.25, 0.5, 1.0]
        else:
            thresholds = [0.1, 0.5, 1.0]

    metrics = {}
    total_samples = len(distances)

    for threshold in thresholds:
        if total_samples == 0:
            robust_acc = 1.0
        else:
            # Samples are robust if attack failed OR distance > threshold
            robust_samples = np.sum(~success_mask) + np.sum(
                distances[success_mask] > threshold
            )
            robust_acc = robust_samples / total_samples

        # Format threshold for key name
        threshold_str = f"{threshold:.3f}".rstrip("0").rstrip(".")
        if threshold == 8 / 255:
            threshold_str = "8_255"  # Safe key name
        elif threshold == 4 / 255:
            threshold_str = "4_255"
        elif threshold == 16 / 255:
            threshold_str = "16_255"

        metrics[f"robust_acc_{threshold_str}"] = float(robust_acc)

    return metrics
