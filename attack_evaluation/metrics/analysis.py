"""
Main analysis functions for attack results.
"""
import numpy as np
from typing import Dict, List, Any, Optional

from .distances import compute_distance_statistics
from .curves import compute_robust_accuracy_curve  
from .storage import save_precompiled_distances


def get_stats(
    attack_data: Dict[str, Any],
    threat_model: str,
    certified_thresholds: Optional[List[float]] = None,
    save_precompiled: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for attack results.
    
    Args:
        attack_data: Results from run_attack()
        threat_model: Threat model used ('linf', 'l2', etc.)
        certified_thresholds: Thresholds for certified robustness
        save_precompiled: Whether to save precompiled distances
        output_dir: Directory to save precompiled data
        
    Returns:
        Dictionary with comprehensive attack statistics
    """
    
    stats = {}
    
    # 1. Distance statistics
    distances = attack_data.get('distances', {})
    for norm, dist_values in distances.items():
        dist_stats = compute_distance_statistics(dist_values)
        for key, value in dist_stats.items():
            stats[f'{norm}_{key}'] = value
    
    # 2. Optimality computation
    if threat_model in distances and threat_model in attack_data.get('best_optim_distances', {}):
        main_distances = distances[threat_model]
        best_distances = attack_data['best_optim_distances'][threat_model]
        optimality = _compute_optimality_score(main_distances, best_distances)
        stats['optimality'] = optimality
    
    # 3. Robust accuracy curves
    if threat_model in distances:
        distances_array = np.array(distances[threat_model])
        success_mask = np.array(attack_data.get('adv_success', []))
        
        curve_data = compute_robust_accuracy_curve(distances_array, success_mask)
        stats['robust_accuracy_curve'] = curve_data
        
        # Certified robustness metrics
        if certified_thresholds is None:
            certified_thresholds = _get_default_thresholds(threat_model)
        
        if certified_thresholds:
            cert_metrics = _compute_certified_metrics(distances_array, success_mask, certified_thresholds)
            stats.update(cert_metrics)
    
    # 4. Save precompiled distances if requested
    if save_precompiled and output_dir:
        precompiled_path = save_precompiled_distances(attack_data, threat_model, output_dir)
        stats['precompiled_path'] = precompiled_path
    
    return stats


def _compute_optimality_score(main_distances: List[float], best_distances: List[float]) -> float:
    """Compute optimality score"""
    main_arr = np.array(main_distances)
    best_arr = np.array(best_distances)
    
    valid_mask = (main_arr > 0) & (best_arr > 0)
    if not valid_mask.any():
        return float('nan')
    
    ratios = main_arr[valid_mask] / best_arr[valid_mask]
    return float(np.mean(ratios))


def _get_default_thresholds(threat_model: str) -> List[float]:
    """Get default certified robustness thresholds by threat model"""
    if threat_model == 'linf':
        return [4/255, 8/255, 16/255]
    elif threat_model == 'l2':
        return [0.5, 1.0, 2.0]
    else:
        return []


def _compute_certified_metrics(distances: np.ndarray, success_mask: np.ndarray, thresholds: List[float]) -> Dict[str, float]:
    """Compute certified robustness metrics for specific thresholds"""
    metrics = {}
    
    for threshold in thresholds:
        robust_mask = (~success_mask) | (distances > threshold)
        robust_acc = robust_mask.mean()
        
        # Convert threshold to readable format
        if threshold < 1:
            threshold_str = f"{threshold*255:.0f}/255"
        else:
            threshold_str = f"{threshold:.3f}"
            
        metrics[f'robust_acc_{threshold_str}'] = float(robust_acc)
    
    return metrics