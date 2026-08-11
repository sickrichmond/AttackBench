"""
Main analysis functions for attack results.
Now computes ALL statistics including basic ones like ASR and accuracy.
"""
from typing import Dict, List, Any, Optional
import numpy as np

from .distances import (compute_distance_statistics, eval_optimality, 
                       compute_attack_efficiency, compute_basic_metrics)
from .curves import (compute_robust_accuracy_curve, compute_auc_robust_accuracy, 
                    compute_certified_robustness_metrics)
from .ensemble import analyze_attack_ensemble
from .optimality import compute_local_optimality, _clean_accuracy

def get_stats(
    attack_results: Dict[str, Any], 
    threat_model: str,
    # Controllo granulare delle analisi
    include_optimality: bool = True,
    include_curves: bool = True, 
    include_certified: bool = True,
    include_efficiency: bool = False,
    # Ottimalità - auto-download da W&B se possibile
    best_distances: Optional[List[float]] = None,
    auto_load_best: bool = True,
    model_name: Optional[str] = None,
    batch_size: Optional[int] = None,
    cache_dir: str = "./cache",
    # Certificazione
    certified_thresholds: Optional[List[float]] = None,
    # Storage
    save_precompiled: bool = False,
    output_dir: Optional[str] = None,
    attack_name: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics from RAW attack results.
    
    This is the main entry point for attack analysis. It computes:
    - Basic metrics (ASR, accuracy) 
    - Distance statistics (mean, median, std, etc.)
    - Optimality scores (via compute_local_optimality with hash-based matching)
    - Robust accuracy curves
    - Certified robustness metrics
    - Query efficiency metrics
    
    Optimality is delegated to ``compute_local_optimality()``, which uses
    hash-based sample matching when downloading optimal distances from W&B.
    This ensures consistent results regardless of sample ordering or subset.
    
    Args:
        attack_results: Raw results from run_attack()
        threat_model: Threat model used ('linf', 'l2', etc.)
        include_optimality: Compute optimality scores
        include_curves: Compute robust accuracy curves  
        include_certified: Compute certified robustness
        include_efficiency: Compute query efficiency
        best_distances: Explicit best known distances for optimality.
            If provided, used directly as reference (positional matching).
            If None and auto_load_best=True, optimal distances are downloaded
            from W&B using hash-based matching.
        auto_load_best: Try to auto-load optimal distances from W&B
        model_name: Model name for W&B lookup
        batch_size: Deprecated, no longer used. Kept for backward compatibility.
        cache_dir: Cache directory for W&B downloads
        certified_thresholds: Custom thresholds for certified robustness
        save_precompiled: Save precompiled distances
        output_dir: Output directory for saved files
        attack_name: Attack name for metadata
        **kwargs: Additional keyword arguments. Supports:
            - dataset (str): Dataset name for W&B lookup (if not in metadata)
        
    Returns:
        Dictionary with comprehensive analysis results
    """
    
    stats = {}
    
    # 0. ALWAYS: Basic metrics (ASR, accuracy) - now computed here!
    basic_metrics = compute_basic_metrics(attack_results)
    stats.update(basic_metrics)
    
    # 1. ALWAYS: Distance statistics for all norms
    distances = attack_results.get('distances', {})
    for norm, dist_values in distances.items():
        if dist_values:
            dist_stats = compute_distance_statistics(dist_values)
            # Add norm prefix to all keys
            for key, value in dist_stats.items():
                stats[f'{norm}_{key}'] = value
    
    # Extract main threat model data
    threat_distances = distances.get(threat_model, [])
    success_mask = np.array(attack_results.get('adv_success', []))
    
    if not threat_distances:
        return stats
    
    threat_distances_array = np.array(threat_distances)
    
    # 2. OPTIONAL: Optimality computation
    if include_optimality:
        # The reference a* is the lower envelope over OTHER attacks: either passed in
        # explicitly, or downloaded from W&B. It is never derived from this attack's own
        # distances — that would compare the attack against itself and score ~1.0 always.
        try:
            reference_results = None
            if best_distances is not None and len(best_distances):
                # Wrap explicit best_distances as a synthetic reference result
                reference_results = [{
                    'distances': {threat_model: list(best_distances)}
                }]

            opt_result = compute_local_optimality(
                attack_results=attack_results,
                reference_results=reference_results,
                threat_model=threat_model,
                dataset=kwargs.get('dataset', None),
                model_name=model_name,
                use_wandb=auto_load_best and reference_results is None,
                cache_dir=cache_dir
            )
            if not np.isnan(opt_result['optimality']):
                stats['optimality'] = opt_result['optimality']
                stats['optimality_reference'] = opt_result['reference_type']
        except Exception as e:
            print(f"Warning: optimality not computed ({e}).\n"
                  f"         Pass best_distances=[...] / reference_results, or make the W&B "
                  f"optimal distances available for this dataset/model/threat model.")
    
    # 3. OPTIONAL: Robust accuracy curves
    if include_curves:
        try:
            curve_data = compute_robust_accuracy_curve(threat_distances_array, success_mask)
            stats['robust_accuracy_curve'] = curve_data
            
            # Compute AUC
            if curve_data['thresholds']:
                auc = compute_auc_robust_accuracy(curve_data['thresholds'], 
                                                curve_data['robust_accuracies'])
                stats['robust_accuracy_auc'] = auc
        except Exception as e:
            print(f"Warning: Could not compute robust accuracy curves: {e}")
    
    # 4. OPTIONAL: Certified robustness metrics
    if include_certified:
        try:
            cert_metrics = compute_certified_robustness_metrics(
                threat_distances_array, success_mask, threat_model, certified_thresholds
            )
            stats.update(cert_metrics)
        except Exception as e:
            print(f"Warning: Could not compute certified robustness: {e}")
    
    # 5. OPTIONAL: Query efficiency
    if include_efficiency:
        num_forwards = attack_results.get('num_forwards', [])
        if num_forwards:
            try:
                efficiency_metrics = compute_attack_efficiency(threat_distances, num_forwards)
                stats.update(efficiency_metrics)
            except Exception as e:
                print(f"Warning: Could not compute efficiency: {e}")
    
    # 6. OPTIONAL: Save precompiled distances
    if save_precompiled and output_dir:
        try:
            from .storage import save_precompiled_distances
            saved_path, _ = save_precompiled_distances(
                attack_results, output_dir=output_dir, threat_model=threat_model,
                model_name=model_name, attack_name=attack_name
            )
            stats['precompiled_path'] = saved_path
        except Exception as e:
            print(f"Warning: Could not save precompiled distances: {e}")
    
    return stats

def compare_attacks(attack_results_list: List[Dict[str, Any]], 
                   threat_model: str,
                   attack_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compare multiple attacks and compute ensemble metrics.
    """
    if len(attack_results_list) < 2:
        return {}
    
    # Add attack names to results
    if attack_names:
        for i, results in enumerate(attack_results_list):
            if i < len(attack_names):
                results['attack_name'] = attack_names[i]
    
    # Individual statistics
    individual_stats = []
    for i, results in enumerate(attack_results_list):
        stats = get_stats(results, threat_model, 
                         include_optimality=False,  # Skip for speed
                         include_curves=False,      # Skip for speed
                         include_certified=True)
        individual_stats.append(stats)
    
    # Ensemble analysis
    ensemble_results = analyze_attack_ensemble(attack_results_list, threat_model)
    
    return {
        'individual_statistics': individual_stats,
        'ensemble_analysis': ensemble_results,
        'threat_model': threat_model
    }

# FUNZIONI PUBBLICHE PER L'UTENTE
def compute_curves(attack_results: Dict[str, Any], threat_model: str) -> Dict[str, Any]:
    """Public function: compute only robust accuracy curves."""
    distances = np.array(attack_results.get('distances', {}).get(threat_model, []))
    success_mask = np.array(attack_results.get('adv_success', []))
    
    if len(distances) == 0:
        return {}
    
    curve_data = compute_robust_accuracy_curve(distances, success_mask)
    auc = compute_auc_robust_accuracy(curve_data['thresholds'], curve_data['robust_accuracies'])
    
    return {
        'robust_accuracy_curve': curve_data,
        'robust_accuracy_auc': auc
    }

def compute_optimality(attack_results: Dict[str, Any], threat_model: str,
                      best_distances: List[float]) -> float:
    """Public function: compute only optimality score."""
    distances = np.array(attack_results.get('distances', {}).get(threat_model, []))

    if len(distances) == 0 or not len(best_distances):
        return float('nan')

    return eval_optimality(distances, best_distances, _clean_accuracy(attack_results))

def compute_efficiency(attack_results: Dict[str, Any], threat_model: str) -> Dict[str, float]:
    """Public function: compute only efficiency metrics."""
    distances = attack_results.get('distances', {}).get(threat_model, [])
    num_forwards = attack_results.get('num_forwards', [])
    
    return compute_attack_efficiency(distances, num_forwards)