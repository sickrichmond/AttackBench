"""
Optimality computation for AttackBench (Stage 3).
Provides user-friendly API to compute local optimality scores.
"""
import numpy as np
from typing import Dict, List, Any, Union, Optional
from .distances import eval_optimality as _eval_optimality_core
from .ensemble import ensemble_distances
from attackbench.wandb_manager import download_precompiled_distances


def compute_local_optimality(
    attack_results: Dict[str, Any],
    reference_results: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
    threat_model: str = 'linf',
    dataset: Optional[str] = None,
    model_name: Optional[str] = None,
    use_wandb: bool = False
) -> Dict[str, float]:
    """
    Compute local optimality score for an attack (Stage 3 of AttackBench).
    
    Optimality measures how close an attack is to the best known distances (ensemble lower bound).
    Score is in [0, 1] where 1.0 = optimal (matches or beats ensemble).
    
    Args:
        attack_results: Output from attackbench.run_attack()
        reference_results: Reference distances for comparison. Can be:
            - Single attack results dict (will use its distances)
            - List of attack results (will compute ensemble)
            - None (will try to download from W&B if use_wandb=True)
        threat_model: Threat model ('linf', 'l0', 'l1', 'l2')
        dataset: Dataset name (required if use_wandb=True)
        model_name: Model name (required if use_wandb=True)
        use_wandb: If True and reference_results is None, download best distances from W&B
        
    Returns:
        Dictionary with:
            - 'optimality': Local optimality score [0, 1]
            - 'auc_attack': Area under curve for the attack
            - 'auc_reference': Area under curve for reference
            - 'reference_type': Type of reference used ('ensemble', 'single', 'wandb')
            
    Example:
        >>> # Compare to another attack
        >>> results_pgd = run_attack(model, dataset, pgd, 'linf', device)
        >>> results_apgd = run_attack(model, dataset, apgd, 'linf', device)
        >>> opt = compute_local_optimality(results_pgd, reference_results=results_apgd)
        >>> print(f"PGD optimality vs APGD: {opt['optimality']:.2%}")
        
        >>> # Compare to ensemble of multiple attacks
        >>> opt = compute_local_optimality(results_pgd, reference_results=[results_apgd, results_df])
        
        >>> # Download best distances from W&B
        >>> opt = compute_local_optimality(results, dataset='cifar10', model_name='Standard', use_wandb=True)
    """
    # Extract attack distances
    attack_distances = np.array(attack_results.get('distances', {}).get(threat_model, []))
    if len(attack_distances) == 0:
        raise ValueError(f"No distances found for threat model '{threat_model}'")
    
    # Determine reference distances
    reference_type = None
    
    if reference_results is not None:
        # User provided reference(s)
        if isinstance(reference_results, dict):
            # Single reference attack
            reference_distances = np.array(reference_results.get('distances', {}).get(threat_model, []))
            reference_type = 'single'
        elif isinstance(reference_results, list):
            # Ensemble of multiple attacks
            if len(reference_results) == 0:
                raise ValueError("reference_results list is empty")
            
            # Compute ensemble (element-wise minimum)
            distances_list = []
            for result in reference_results:
                dist = np.array(result.get('distances', {}).get(threat_model, []))
                if len(dist) > 0:
                    distances_list.append(dist)
            
            if len(distances_list) == 0:
                raise ValueError("No valid distances in reference_results")
            
            # Start with first attack and iteratively compute minimum
            reference_distances = distances_list[0]
            for dist in distances_list[1:]:
                reference_distances = ensemble_distances(reference_distances, dist)
            
            reference_type = f'ensemble_{len(distances_list)}'
        else:
            raise TypeError("reference_results must be dict or list of dicts")
            
    elif use_wandb:
        # Download from W&B
        if dataset is None or model_name is None:
            raise ValueError("dataset and model_name required when use_wandb=True")
        
        # Try to find best distances on W&B
        # This will search for the latest artifact for this scenario
        try:
            best_data = download_precompiled_distances(
                dataset=dataset,
                threat_model=threat_model,
                model_name=model_name,
                attack_name='ensemble',  # Try to find ensemble first
                n_samples=len(attack_distances)
            )
            reference_distances = np.array(best_data.get('distances', {}).get(threat_model, []))
            reference_type = 'wandb_ensemble'
        except Exception:
            # If ensemble not found, could try individual attacks
            # For now, just raise an error
            raise ValueError(f"No ensemble distances found on W&B for {dataset}/{model_name}/{threat_model}")
    else:
        raise ValueError("Must provide reference_results or set use_wandb=True")
    
    # Validate reference distances
    if len(reference_distances) == 0:
        raise ValueError("Reference distances are empty")
    if len(reference_distances) != len(attack_distances):
        raise ValueError(f"Length mismatch: attack has {len(attack_distances)} samples, "
                       f"reference has {len(reference_distances)} samples")
    
    # Compute optimality using core function
    optimality = _eval_optimality_core(attack_distances, reference_distances)
    
    # Compute AUC for both
    auc_attack = _compute_auc(attack_distances)
    auc_reference = _compute_auc(reference_distances)
    
    return {
        'optimality': optimality,
        'auc_attack': auc_attack,
        'auc_reference': auc_reference,
        'reference_type': reference_type,
        'n_samples': len(attack_distances)
    }


def _compute_auc(distances: np.ndarray) -> float:
    """Helper to compute AUC under robust accuracy curve."""
    unique_distances, counts = np.unique(distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(distances)
    auc = np.trapz(robust_acc, unique_distances)
    return float(auc)


def compare_attacks_optimality(
    attack_results_list: List[Dict[str, Any]],
    threat_model: str = 'linf',
    attack_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compare optimality of multiple attacks against their ensemble.
    
    Args:
        attack_results_list: List of attack results from run_attack()
        threat_model: Threat model to analyze
        attack_names: Optional names for attacks (default: Attack_0, Attack_1, ...)
        
    Returns:
        Dictionary with:
            - 'optimality_scores': Dict mapping attack names to optimality scores
            - 'ensemble_distances': The ensemble (best) distances used as reference
            - 'ranking': List of (name, score) tuples sorted by optimality
            
    Example:
        >>> results_list = [results_pgd, results_apgd, results_deepfool]
        >>> comparison = compare_attacks_optimality(results_list, attack_names=['PGD', 'APGD', 'DeepFool'])
        >>> for name, score in comparison['ranking']:
        >>>     print(f"{name}: {score:.2%}")
    """
    if len(attack_results_list) < 2:
        raise ValueError("Need at least 2 attacks to compare")
    
    # Generate names if not provided
    if attack_names is None:
        attack_names = [f"Attack_{i}" for i in range(len(attack_results_list))]
    
    if len(attack_names) != len(attack_results_list):
        raise ValueError("Length of attack_names must match attack_results_list")
    
    # Extract all distances
    distances_list = []
    for results in attack_results_list:
        dist = np.array(results.get('distances', {}).get(threat_model, []))
        if len(dist) == 0:
            raise ValueError(f"No distances found for threat model '{threat_model}'")
        distances_list.append(dist)
    
    # Compute ensemble (element-wise minimum across all attacks)
    ensemble = distances_list[0].copy()
    for dist in distances_list[1:]:
        ensemble = ensemble_distances(ensemble, dist)
    
    # Compute optimality for each attack
    optimality_scores = {}
    for name, distances in zip(attack_names, distances_list):
        opt = _eval_optimality_core(distances, ensemble)
        optimality_scores[name] = opt
    
    # Create ranking (sorted by optimality, descending)
    ranking = sorted(optimality_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'optimality_scores': optimality_scores,
        'ensemble_distances': ensemble,
        'ranking': ranking,
        'n_attacks': len(attack_results_list),
        'n_samples': len(ensemble)
    }
