"""
Optimality computation for AttackBench (Stage 3).
Provides user-friendly API to compute local optimality scores.
"""
import numpy as np
from typing import Dict, List, Any, Union, Optional
from .distances import eval_optimality as _eval_optimality_core
from .ensemble import ensemble_distances
# NOTE: download_optimal_distances is imported lazily inside functions to avoid circular import

# NumPy 2.0+ compatibility: trapz was renamed to trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


def compute_local_optimality(
    attack_results: Dict[str, Any],
    reference_results: Optional[List[Dict[str, Any]]] = None,
    threat_model: Optional[str] = None,
    dataset: Optional[str] = None,
    model_name: Optional[str] = None,
    use_wandb: bool = True,
    cache_dir: str = "./cache"
) -> Dict[str, float]:
    """
    Compute local optimality score for an attack (Stage 3 of AttackBench).
    
    Optimality measures how close an attack is to the best known distances (lower envelope).
    Score is in [0, 1] where 1.0 = optimal (matches or beats the lower envelope).
    
    Metadata (dataset, model_name, threat_model, n_samples) is automatically extracted
    from attack_results['metadata'] if available from run_attack().
    
    Args:
        attack_results: Output from attackbench.run_attack() - must contain 'metadata' block
        reference_results: Optional list of attack results to compute ensemble as reference.
            If provided, computes element-wise minimum (lower envelope) from these.
            If None, downloads optimal distances from W&B (if use_wandb=True).
        threat_model: Override threat model (default: extract from attack_results['metadata'])
        dataset: Override dataset name (default: extract from attack_results['metadata'])
        model_name: Override model name (default: extract from attack_results['metadata'])
        use_wandb: If True and reference_results is None, download optimal distances from W&B
        cache_dir: Local cache directory for W&B downloads
        
    Returns:
        Dictionary with:
            - 'optimality': Local optimality score [0, 1]
            - 'auc_attack': Area under curve for the attack
            - 'auc_reference': Area under curve for reference (lower envelope)
            - 'reference_type': Type of reference used ('ensemble_N', 'wandb_optimal')
            - 'n_samples': Number of samples evaluated
            
    Example:
        >>> # Automatic: uses metadata from run_attack and downloads optimal from W&B
        >>> results = run_attack(model, dataset, pgd, 'linf', device,
        ...                      dataset_name='cifar10', model_name='Standard')
        >>> opt = compute_local_optimality(results)
        >>> print(f"Optimality: {opt['optimality']:.2%}")
        
        >>> # Compare to ensemble of multiple attacks (computes lower envelope locally)
        >>> opt = compute_local_optimality(results_pgd, reference_results=[results_apgd, results_df])
    """
    # Extract metadata from attack_results (populated by run_attack)
    metadata = attack_results.get('metadata', {})
    
    # Use metadata or override with explicit parameters
    threat_model = threat_model or metadata.get('threat_model')
    dataset = dataset or metadata.get('dataset')
    model_name = model_name or metadata.get('model_name')
    
    if threat_model is None:
        raise ValueError(
            "threat_model not found. Either pass it explicitly or ensure attack_results "
            "contains 'metadata' block from run_attack(threat_model=...)"
        )
    
    # Extract attack distances
    attack_distances = np.array(attack_results.get('distances', {}).get(threat_model, []))
    if len(attack_distances) == 0:
        raise ValueError(f"No distances found for threat model '{threat_model}'")
    
    n_samples = len(attack_distances)
    reference_type = None
    
    # Determine reference distances (lower envelope)
    if reference_results is not None:
        # User provided list of attacks - compute ensemble (lower envelope)
        if not isinstance(reference_results, list):
            raise TypeError(
                "reference_results must be a list of attack results. "
                "Single attack comparison is not supported - use a list of attacks "
                "to compute the lower envelope, or set use_wandb=True to download "
                "optimal distances from W&B."
            )
        
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
        
    elif use_wandb:
        # Download optimal distances (lower envelope) from W&B
        if dataset is None or model_name is None:
            raise ValueError(
                "dataset and model_name required to download optimal distances from W&B. "
                "Either pass them explicitly or ensure attack_results contains 'metadata' "
                "block from run_attack(dataset_name=..., model_name=...)."
            )
        
        # Lazy import to avoid circular dependency
        from attackbench.wandb_manager import download_optimal_distances
        
        # Download optimal distances artifact
        best_data = download_optimal_distances(
            dataset=dataset,
            threat_model=threat_model,
            model_name=model_name,
            n_samples=n_samples,
            cache_dir=cache_dir
        )
        
        if best_data is None:
            raise ValueError(
                f"No optimal distances found on W&B for {dataset}/{threat_model}/{model_name}/{n_samples}. "
                f"Upload optimal distances first using upload_optimal_distances()."
            )
        
        reference_distances = np.array(best_data.get('distances', {}).get(threat_model, []))
        if len(reference_distances) == 0:
            raise ValueError(f"Downloaded data does not contain distances for threat model '{threat_model}'")
        
        reference_type = 'wandb_optimal'
    else:
        raise ValueError(
            "Must provide reference_results (list of attacks for ensemble) or set use_wandb=True "
            "to download optimal distances from W&B."
        )
    
    # Validate reference distances
    if len(reference_distances) == 0:
        raise ValueError("Reference distances are empty")
    if len(reference_distances) != n_samples:
        raise ValueError(f"Length mismatch: attack has {n_samples} samples, "
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
    auc = _trapz(robust_acc, unique_distances)
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
