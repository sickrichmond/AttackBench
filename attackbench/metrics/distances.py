"""
Distance computation and statistics for attack results.
Includes optimality computation using best precompiled distances.
"""
import warnings

import numpy as np
from typing import Dict, List, Any, Optional


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


def _aurec(distances, eps_0: float, n_total: int) -> float:
    """
    Area under the robustness evaluation curve, AUREC(eps_0) = int_0^eps_0 rho(eps) deps
    (AttackBench Eq. 4), where rho(eps) = 1 - ASR(eps) is the robust accuracy.

    rho is a right-continuous step function that drops by 1/n_total at every perturbation
    size found by the attack, so the integral is an exact sum of rectangles — using the
    trapezoidal rule here would systematically bias the area.

    Distances above eps_0 (and inf, i.e. samples the attack never broke) do not lower the
    curve inside the integration window, but they stay in n_total: that is what keeps the
    curve high for attacks that fail.
    """
    d = np.sort(np.asarray(distances, dtype=float))
    d = d[np.isfinite(d) & (d <= eps_0)]

    rho = 1.0 - np.arange(1, len(d) + 1) / n_total
    edges = np.concatenate(([0.0], d, [eps_0]))
    heights = np.concatenate(([1.0], rho))
    return float(np.sum(heights * np.diff(edges)))


def eval_optimality(adv_distances, best_distances, clean_acc: Optional[float] = None) -> float:
    """
    Local optimality of an attack against a reference lower envelope (AttackBench Eq. 5):

        LO = (rho * eps_0 - AUREC_attack) / (rho * eps_0 - AUREC_reference)

    where rho is the clean accuracy of the target model and eps_0 is the smallest
    perturbation size at which the reference curve reaches zero robust accuracy.

    Args:
        adv_distances: per-sample distances of the attack (d*, not the last iterate)
        best_distances: per-sample distances of the empirical best attack a* (lower envelope)
        clean_acc: clean accuracy of the model. If None it is estimated as the fraction of
            reference distances that are non-zero (samples already misclassified have
            distance 0 by convention) — pass the real value when you have it.

    Returns:
        Optimality in [0, 1]: 1.0 = as good as the lower envelope, 0.0 = attack never
        beats the model. NaN if the reference has no usable spread.
    """
    adv_distances = np.asarray(adv_distances, dtype=float)
    best_distances = np.asarray(best_distances, dtype=float)

    if len(best_distances) == 0 or len(adv_distances) == 0:
        return float('nan')

    n_total = len(best_distances)
    if clean_acc is None:
        clean_acc = float(np.count_nonzero(best_distances) / n_total)

    finite_ref = best_distances[np.isfinite(best_distances)]
    if len(finite_ref) == 0:
        return float('nan')
    eps_0 = float(finite_ref.max())

    if len(finite_ref) < n_total:
        warnings.warn(
            f'The reference lower envelope leaves {n_total - len(finite_ref)}/{n_total} samples '
            f'unbroken, so its robust accuracy never reaches 0. eps_0 is set to the largest '
            f'finite reference distance ({eps_0:.4g}); optimality scores stay comparable '
            f'between attacks but are not the paper-defined eps_0.'
        )

    aurec_ref = _aurec(best_distances, eps_0, n_total)
    aurec_attack = _aurec(adv_distances, eps_0, len(adv_distances))

    denominator = clean_acc * eps_0 - aurec_ref
    if denominator <= 0:
        # Degenerate: the reference has no spread (all distances equal or all zero).
        return float('nan')

    return float(np.clip((clean_acc * eps_0 - aurec_attack) / denominator, 0.0, 1.0))


def compute_basic_metrics(attack_results: Dict[str, Any]) -> Dict[str, float]:
    """Compute ASR and clean accuracy from raw attack results."""
    adv_success = attack_results.get('adv_success', [])
    ori_success = attack_results.get('ori_success', [])
    correct = attack_results.get('correct', [])

    metrics = {}

    if len(adv_success) > 0:
        metrics['ASR'] = sum(adv_success) / len(adv_success)

    # ori_success is "already misclassified before the attack", i.e. the complement of
    # clean correctness — reporting it as accuracy inverts the metric.
    if len(correct) > 0:
        metrics['accuracy'] = sum(correct) / len(correct)
    elif len(ori_success) > 0:
        metrics['accuracy'] = 1 - sum(ori_success) / len(ori_success)

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