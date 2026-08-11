"""
Optimality computation for AttackBench (Stage 3).
Provides user-friendly API to compute local optimality scores.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from .distances import eval_optimality as _eval_optimality_core, _aurec
# NOTE: download_optimal_distances is imported lazily inside functions to avoid circular import


def _clean_accuracy(attack_results: Dict[str, Any]) -> Optional[float]:
    """Clean accuracy of the target model, from the per-sample correctness flags."""
    correct = attack_results.get('correct', [])
    if len(correct):
        return float(np.mean(correct))
    ori_success = attack_results.get('ori_success', [])
    if len(ori_success):
        return 1.0 - float(np.mean(ori_success))
    return None


def _lower_envelope(distances_list: List[np.ndarray]) -> np.ndarray:
    """Sample-wise minimum across attacks — the empirical best attack a* of the paper."""
    return np.minimum.reduce(distances_list)


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

        reference_distances = _lower_envelope(distances_list)
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
        from ..wandb.manager import download_optimal_distances
        
        # Download optimal distances artifact (hash-based, full dataset)
        best_data = download_optimal_distances(
            dataset=dataset,
            threat_model=threat_model,
            model_name=model_name,
            cache_dir=cache_dir
        )
        
        if best_data is None:
            raise ValueError(
                f"Could not load optimal distances for {dataset}/{threat_model}/{model_name}. "
                "This can happen if:\n"
                "  1. W&B credentials are not configured (run 'wandb login' or set WANDB_API_KEY)\n"
                "  2. The optimal distances artifact doesn't exist on W&B\n"
                "Alternatively, compute optimality locally by passing reference_results="
                "(list of attack results to use as reference)."
            )
        
        optimal_distances_data = best_data.get('distances', {}).get(threat_model, {})
        
        if isinstance(optimal_distances_data, dict):
            # Hash-based format: {hash: distance, ...}
            # Match attack samples by hash
            attack_hashes = attack_results.get('hashes', [])
            if not attack_hashes:
                raise ValueError(
                    "attack_results does not contain 'hashes'. Hashes are required to match "
                    "samples against hash-based optimal distances. Re-run the attack with "
                    "a recent version of AttackBench that always includes hashes."
                )
            
            reference_distances_list = []
            missing_hashes = []
            for h in attack_hashes:
                if h in optimal_distances_data:
                    reference_distances_list.append(optimal_distances_data[h])
                else:
                    missing_hashes.append(h)
            
            if missing_hashes:
                raise ValueError(
                    f"{len(missing_hashes)}/{len(attack_hashes)} attack sample hashes not found "
                    f"in optimal distances. The optimal distances may have been computed on a "
                    f"different dataset or dataset version. First missing hash: {missing_hashes[0][:16]}..."
                )
            
            reference_distances = np.array(reference_distances_list)
        elif isinstance(optimal_distances_data, list):
            # Legacy positional format: [distance, distance, ...]
            reference_distances = np.array(optimal_distances_data)
        else:
            raise ValueError(f"Unexpected optimal distances format for threat model '{threat_model}'")
        
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
                       f"reference has {len(reference_distances)} samples. "
                       f"Ensure the reference was computed on compatible samples.")
    
    # Compute optimality using core function
    clean_acc = _clean_accuracy(attack_results)
    optimality = _eval_optimality_core(attack_distances, reference_distances, clean_acc)

    # Areas under the robustness evaluation curves, both integrated over [0, eps_0]
    finite_ref = reference_distances[np.isfinite(reference_distances)]
    eps_0 = float(finite_ref.max()) if len(finite_ref) else float('nan')
    auc_attack = _aurec(attack_distances, eps_0, n_samples) if len(finite_ref) else float('nan')
    auc_reference = _aurec(reference_distances, eps_0, n_samples) if len(finite_ref) else float('nan')

    return {
        'optimality': optimality,
        'auc_attack': auc_attack,
        'auc_reference': auc_reference,
        'reference_type': reference_type,
        'n_samples': len(attack_distances)
    }


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
    ensemble = _lower_envelope(distances_list)

    # Compute optimality for each attack
    optimality_scores = {}
    for name, results, distances in zip(attack_names, attack_results_list, distances_list):
        optimality_scores[name] = _eval_optimality_core(
            distances, ensemble, _clean_accuracy(results)
        )
    
    # Create ranking (sorted by optimality, descending)
    ranking = sorted(optimality_scores.items(), key=lambda x: x[1], reverse=True)
    
    return {
        'optimality_scores': optimality_scores,
        'ensemble_distances': ensemble,
        'ranking': ranking,
        'n_attacks': len(attack_results_list),
        'n_samples': len(ensemble)
    }
