"""
Robust accuracy curves and evaluation metrics.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def compute_robust_accuracy_curve(distances: np.ndarray, success_mask: np.ndarray, 
                                  num_points: int = 100) -> Dict[str, List[float]]:
    """
    Compute robust accuracy curve from attack distances and success indicators.
    
    Args:
        distances: Array of attack distances
        success_mask: Boolean array indicating attack success
        num_points: Number of points in the curve
        
    Returns:
        Dictionary with thresholds and robust accuracies
    """
    if len(distances) != len(success_mask):
        raise ValueError("Distances and success mask must have same length")
    
    if len(distances) == 0:
        return {'thresholds': [], 'robust_accuracies': []}
    
    # Get range of thresholds based on successful attack distances
    successful_distances = distances[success_mask]
    
    if len(successful_distances) == 0:
        # No successful attacks - perfect robustness
        max_distance = np.max(distances) if len(distances) > 0 else 1.0
        thresholds = np.linspace(0, max_distance, num_points)
        robust_accuracies = [1.0] * num_points
    else:
        # Create thresholds from 0 to max successful distance
        min_threshold = 0.0
        max_threshold = np.max(successful_distances) * 1.1  # Add 10% margin
        
        thresholds = np.linspace(min_threshold, max_threshold, num_points)
        robust_accuracies = []
        
        for threshold in thresholds:
            # Robust at this threshold if:
            # 1. Attack failed (not in success_mask), OR
            # 2. Attack succeeded but distance > threshold
            robust_samples = (~success_mask) | (distances > threshold)
            robust_accuracy = robust_samples.mean()
            robust_accuracies.append(float(robust_accuracy))
    
    return {
        'thresholds': thresholds.tolist(),
        'robust_accuracies': robust_accuracies
    }


def compute_auc_robust_accuracy(thresholds: List[float], robust_accuracies: List[float], 
                               max_threshold: Optional[float] = None) -> float:
    """
    Compute Area Under Curve for robust accuracy.
    
    Args:
        thresholds: List of threshold values
        robust_accuracies: List of robust accuracy values
        max_threshold: Maximum threshold to consider (None for all)
        
    Returns:
        AUC value
    """
    if len(thresholds) != len(robust_accuracies):
        raise ValueError("Thresholds and accuracies must have same length")
    
    if len(thresholds) < 2:
        return 0.0
    
    thresholds = np.array(thresholds)
    robust_accuracies = np.array(robust_accuracies)
    
    # Filter by max_threshold if specified
    if max_threshold is not None:
        mask = thresholds <= max_threshold
        thresholds = thresholds[mask]
        robust_accuracies = robust_accuracies[mask]
    
    if len(thresholds) < 2:
        return 0.0
    
    # Compute AUC using trapezoidal rule
    auc = float(np.trapz(robust_accuracies, thresholds))
    
    # Normalize by threshold range
    threshold_range = thresholds[-1] - thresholds[0]
    if threshold_range > 0:
        auc = auc / threshold_range
    
    return auc


def compute_robustness_at_thresholds(distances: np.ndarray, success_mask: np.ndarray,
                                   thresholds: List[float]) -> Dict[str, float]:
    """
    Compute robust accuracy at specific threshold values.
    
    Args:
        distances: Array of attack distances
        success_mask: Boolean array indicating attack success
        thresholds: List of threshold values to evaluate
        
    Returns:
        Dictionary mapping threshold strings to robust accuracies
    """
    if len(distances) != len(success_mask):
        raise ValueError("Distances and success mask must have same length")
    
    results = {}
    
    for threshold in thresholds:
        # Robust samples: attack failed OR distance > threshold
        robust_samples = (~success_mask) | (distances > threshold)
        robust_accuracy = float(robust_samples.mean())
        
        # Format threshold for dictionary key
        if threshold < 1:
            threshold_key = f"{threshold*255:.0f}/255"
        else:
            threshold_key = f"{threshold:.3f}"
            
        results[f"robust_acc_{threshold_key}"] = robust_accuracy
    
    return results


def compute_security_curve_comparison(curves_data: List[Dict[str, Any]], 
                                    attack_names: List[str]) -> Dict[str, Any]:
    """
    Compare multiple security evaluation curves.
    
    Args:
        curves_data: List of curve dictionaries with 'thresholds' and 'robust_accuracies'
        attack_names: Names of the attacks
        
    Returns:
        Dictionary with comparison statistics
    """
    if len(curves_data) != len(attack_names):
        raise ValueError("Number of curves must match number of attack names")
    
    if not curves_data:
        return {}
    
    # Find common threshold range
    all_thresholds = []
    for curve in curves_data:
        all_thresholds.extend(curve['thresholds'])
    
    if not all_thresholds:
        return {}
    
    min_threshold = min(all_thresholds)
    max_threshold = max(all_thresholds)
    
    # Create common threshold grid
    common_thresholds = np.linspace(min_threshold, max_threshold, 100)
    
    # Interpolate all curves to common grid
    interpolated_curves = {}
    auc_scores = {}
    
    for i, (curve, name) in enumerate(zip(curves_data, attack_names)):
        if len(curve['thresholds']) < 2:
            continue
            
        # Interpolate to common grid
        interpolated_acc = np.interp(common_thresholds, curve['thresholds'], curve['robust_accuracies'])
        interpolated_curves[name] = interpolated_acc.tolist()
        
        # Compute AUC
        auc_scores[name] = compute_auc_robust_accuracy(curve['thresholds'], curve['robust_accuracies'])
    
    # Find dominant attack (lowest robust accuracy curve)
    if interpolated_curves:
        min_robust_acc = np.inf
        strongest_attack = None
        
        for name, acc_curve in interpolated_curves.items():
            mean_acc = np.mean(acc_curve)
            if mean_acc < min_robust_acc:
                min_robust_acc = mean_acc
                strongest_attack = name
    else:
        strongest_attack = None
    
    return {
        'common_thresholds': common_thresholds.tolist(),
        'interpolated_curves': interpolated_curves,
        'auc_scores': auc_scores,
        'strongest_attack': strongest_attack,
        'threshold_range': (min_threshold, max_threshold),
    }


def compute_certified_robustness_metrics(distances: np.ndarray, success_mask: np.ndarray,
                                       threat_model: str, custom_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
    """
    Compute certified robustness metrics for standard thresholds.
    
    Args:
        distances: Array of attack distances
        success_mask: Boolean array indicating attack success
        threat_model: Threat model ('linf', 'l2', etc.)
        custom_thresholds: Custom threshold values (overrides defaults)
        
    Returns:
        Dictionary with certified robustness metrics
    """
    if custom_thresholds is not None:
        thresholds = custom_thresholds
    else:
        # Default thresholds based on threat model
        if threat_model == 'linf':
            thresholds = [1/255, 2/255, 4/255, 8/255, 16/255]
        elif threat_model == 'l2':
            thresholds = [0.1, 0.25, 0.5, 1.0, 2.0]
        elif threat_model == 'l1':
            thresholds = [1.0, 5.0, 10.0, 20.0, 50.0]
        elif threat_model == 'l0':
            thresholds = [1, 5, 10, 20, 50]
        else:
            # Generic thresholds
            max_dist = np.max(distances) if len(distances) > 0 else 1.0
            thresholds = [max_dist * r for r in [0.1, 0.2, 0.3, 0.4, 0.5]]
    
    return compute_robustness_at_thresholds(distances, success_mask, thresholds)


def compute_attack_strength_ranking(attack_results: List[Dict[str, Any]], 
                                  attack_names: List[str],
                                  threat_model: str) -> Dict[str, Any]:
    """
    Rank attacks by their strength based on multiple metrics.
    
    Args:
        attack_results: List of attack result dictionaries
        attack_names: Names of the attacks
        threat_model: Threat model used
        
    Returns:
        Dictionary with ranking information
    """
    if len(attack_results) != len(attack_names):
        raise ValueError("Number of results must match number of attack names")
    
    rankings = {}
    metrics_data = {}
    
    # Extract key metrics for each attack
    for i, (results, name) in enumerate(zip(attack_results, attack_names)):
        asr = results.get('ASR', 0.0)
        
        # Get distances for this threat model
        distances = results.get('distances', {}).get(threat_model, [])
        mean_distance = np.mean(distances) if distances else float('inf')
        
        # Get robustness curve AUC if available
        curve_data = results.get('robust_accuracy_curve', {})
        if curve_data and 'thresholds' in curve_data:
            auc = compute_auc_robust_accuracy(curve_data['thresholds'], curve_data['robust_accuracies'])
        else:
            auc = 1.0  # Perfect robustness if no curve
        
        metrics_data[name] = {
            'ASR': asr,
            'mean_distance': mean_distance,
            'auc_robust_acc': auc,
        }
    
    # Rank by ASR (higher is stronger)
    asr_ranking = sorted(attack_names, key=lambda x: metrics_data[x]['ASR'], reverse=True)
    
    # Rank by mean distance (lower is stronger, but only for successful attacks)
    valid_distances = {name: data['mean_distance'] for name, data in metrics_data.items() 
                      if data['mean_distance'] != float('inf')}
    if valid_distances:
        distance_ranking = sorted(valid_distances.keys(), key=lambda x: valid_distances[x])
    else:
        distance_ranking = attack_names
    
    # Rank by AUC (lower is stronger)
    auc_ranking = sorted(attack_names, key=lambda x: metrics_data[x]['auc_robust_acc'])
    
    # Compute composite score (lower is stronger)
    composite_scores = {}
    for name in attack_names:
        # Combine ASR (higher better), distance (lower better), AUC (lower better)
        asr_score = 1.0 - metrics_data[name]['ASR']  # Invert so lower is better
        
        if metrics_data[name]['mean_distance'] != float('inf'):
            # Normalize distance by maximum
            max_dist = max(d for d in [metrics_data[n]['mean_distance'] for n in attack_names] 
                          if d != float('inf'))
            dist_score = metrics_data[name]['mean_distance'] / max_dist if max_dist > 0 else 0
        else:
            dist_score = 1.0  # Worst score for failed attacks
        
        auc_score = metrics_data[name]['auc_robust_acc']
        
        # Weighted combination
        composite_scores[name] = 0.4 * asr_score + 0.3 * dist_score + 0.3 * auc_score
    
    composite_ranking = sorted(attack_names, key=lambda x: composite_scores[x])
    
    return {
        'asr_ranking': asr_ranking,
        'distance_ranking': distance_ranking,
        'auc_ranking': auc_ranking,
        'composite_ranking': composite_ranking,
        'metrics_data': metrics_data,
        'composite_scores': composite_scores,
        'strongest_attack': composite_ranking[0] if composite_ranking else None,
    }