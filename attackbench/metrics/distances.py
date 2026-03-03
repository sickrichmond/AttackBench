"""
Distance computation and statistics for attack results.
Includes optimality computation using best precompiled distances.
"""
import numpy as np
from typing import Dict, List, Any, Optional

# NumPy 2.0+ compatibility: trapz was renamed to trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def compute_distance_statistics(distances: List[float]) -> Dict[str, float]:
    """Compute comprehensive distance statistics from raw distance data."""
    if not distances:
        return {
            'mean_distance': 0.0,
            'median_distance': 0.0,
            'std_distance': 0.0,
            'min_distance': 0.0,
            'max_distance': 0.0,
            'p95_distance': 0.0,
            'p99_distance': 0.0,
            'success_rate': 0.0
        }
    
    distances_array = np.array(distances)
    valid_distances = distances_array[distances_array < float('inf')]
    
    if len(valid_distances) == 0:
        return {k: 0.0 for k in ['mean_distance', 'median_distance', 'std_distance', 
                                'min_distance', 'max_distance', 'p95_distance', 'p99_distance', 'success_rate']}
    
    return {
        'mean_distance': float(np.mean(valid_distances)),
        'median_distance': float(np.median(valid_distances)),
        'std_distance': float(np.std(valid_distances)),
        'min_distance': float(np.min(valid_distances)),
        'max_distance': float(np.max(valid_distances)),
        'p95_distance': float(np.percentile(valid_distances, 95)),
        'p99_distance': float(np.percentile(valid_distances, 99)),
        'success_rate': float(len(valid_distances) / len(distances))
    }


def eval_optimality(adv_distances: np.ndarray, best_distances: list) -> float:
    """
    Compute optimality score comparing attack distances to best known distances.
    Uses AUC-based method from analysis/utils.py (proven implementation).
    
    Args:
        adv_distances: Adversarial distances from attack
        best_distances: Best known distances (reference/optimal)
        
    Returns:
        Optimality score (1.0 = optimal, lower = worse)
    """
    # Convert to numpy arrays if needed
    if isinstance(adv_distances, list):
        adv_distances = np.array(adv_distances)
    if isinstance(best_distances, list):
        best_distances = np.array(best_distances)
    
    # Compute robust accuracy curve for best distances
    distances, counts = np.unique(best_distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(best_distances)

    # Get quantities for optimality calculation
    clean_acc = np.count_nonzero(best_distances) / len(best_distances)
    max_dist = np.amax(distances)
    best_area = _trapz(robust_acc, distances)

    # Compute robust accuracy for attack distances (not used directly)
    distances, counts = np.unique(adv_distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(adv_distances)

    # Clip distances to max_dist for fair comparison
    distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
    robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

    # Calculate area under clipped curve
    area = _trapz(robust_acc_clipped, distances_clipped)
    
    # Calculate optimality (normalized AUC difference)
    optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

    return float(optimality)


def compute_basic_metrics(attack_results: Dict[str, Any]) -> Dict[str, float]:
    """Compute ASR and accuracy from raw attack results."""
    adv_success = attack_results.get('adv_success', [])
    ori_success = attack_results.get('ori_success', [])
    
    metrics = {}
    
    if len(adv_success) > 0:
        metrics['ASR'] = sum(adv_success) / len(adv_success)
        
    if len(ori_success) > 0:
        metrics['accuracy'] = sum(ori_success) / len(ori_success)
    
    return metrics


def compute_attack_efficiency(distances: List[float], num_queries: List[int]) -> Dict[str, float]:
    """Compute query efficiency metrics."""
    if not distances or not num_queries:
        return {}
    
    valid_mask = np.array(distances) < float('inf')
    if not valid_mask.any():
        return {'efficiency_score': 0.0}
    
    valid_distances = np.array(distances)[valid_mask]
    valid_queries = np.array(num_queries)[valid_mask]
    
    # Efficiency = success rate / average queries
    success_rate = len(valid_distances) / len(distances)
    avg_queries = np.mean(valid_queries) if len(valid_queries) > 0 else float('inf')
    
    efficiency = success_rate / avg_queries if avg_queries > 0 else 0.0
    
    return {
        'efficiency_score': float(efficiency),
        'avg_queries_success': float(avg_queries),
        'success_rate': float(success_rate)
    }


# Alias for backward compatibility
compute_optimality_score = eval_optimality