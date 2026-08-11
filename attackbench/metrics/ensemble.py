"""
Ensemble analysis for multiple attacks.
Ported from analysis/utils.py
"""
import numpy as np
from typing import List, Dict, Any

def ensemble_distances(atk1_distances: np.ndarray, atk2_distances: np.ndarray) -> np.ndarray:
    """Compute ensemble distances (element-wise minimum). From analysis/utils.py"""
    return np.minimum(atk1_distances, atk2_distances)

def complementarity(atk1: np.ndarray, atk2: np.ndarray) -> float:
    """
    Compute complementarity between two attacks. From analysis/utils.py
    """
    diversity = (atk1 ^ atk2).sum()
    double_fault = ((atk1 + atk2) == 0).sum()
    available = diversity + double_fault
    
    if available == 0:
        return 0.0
    
    return float(diversity / available)

def ensemble_gain(atk1: np.ndarray, atk2: np.ndarray) -> float:
    """
    Ensemble gain: fraction of samples that atk2 breaks and atk1 does not, i.e. what
    adding atk2 to atk1 buys you.

    Args:
        atk1, atk2: per-sample success flags (1/True = the attack succeeded)
    """
    atk1 = np.asarray(atk1).astype(bool)
    atk2 = np.asarray(atk2).astype(bool)
    if len(atk1) == 0:
        return 0.0

    return float((atk2 & ~atk1).sum() / len(atk1))

def analyze_attack_ensemble(attack_results: List[Dict[str, Any]], 
                          threat_model: str) -> Dict[str, Any]:
    """
    Comprehensive ensemble analysis for multiple attacks.
    """
    if len(attack_results) < 2:
        return {}
    
    # Extract success arrays and distances
    success_arrays = []
    distances_arrays = []
    attack_names = []
    
    for i, results in enumerate(attack_results):
        success = np.array(results.get('adv_success', []))
        distances = np.array(results.get('distances', {}).get(threat_model, []))
        name = results.get('attack_name', f'Attack_{i}')
        
        success_arrays.append(success)
        distances_arrays.append(distances)
        attack_names.append(name)
    
    # Pairwise analysis
    pairwise_metrics = {}
    ensemble_distances_all = distances_arrays[0].copy()
    
    for i in range(len(attack_results)):
        for j in range(i + 1, len(attack_results)):
            pair_key = f"{attack_names[i]}_vs_{attack_names[j]}"
            
            pairwise_metrics[pair_key] = {
                'complementarity': complementarity(success_arrays[i], success_arrays[j]),
                'gain_i_to_j': ensemble_gain(success_arrays[i], success_arrays[j]),
                'gain_j_to_i': ensemble_gain(success_arrays[j], success_arrays[i])
            }
        
        # Update ensemble distances
        if i > 0:
            ensemble_distances_all = ensemble_distances(ensemble_distances_all, distances_arrays[i])
    
    # Overall ensemble metrics
    ensemble_success = np.zeros_like(success_arrays[0])
    for success_arr in success_arrays:
        ensemble_success = ensemble_success | success_arr
    
    ensemble_asr = np.mean(ensemble_success)
    
    return {
        'pairwise_analysis': pairwise_metrics,
        'ensemble_asr': float(ensemble_asr),
        'ensemble_distances': ensemble_distances_all.tolist(),
        'individual_asrs': [float(np.mean(succ)) for succ in success_arrays],
        'attack_names': attack_names
    }