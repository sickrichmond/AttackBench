"""
Global Optimality and Ranking (Stage 4-5 of AttackBench).
Provides user-friendly API for multi-model evaluation and leaderboard generation.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from .optimality import compute_local_optimality, _compute_auc
from .ensemble import ensemble_distances


def compute_global_optimality(
    attack_results_per_model: Dict[str, Dict[str, Any]],
    threat_model: str = 'linf',
    reference_per_model: Optional[Dict[str, Union[Dict[str, Any], List[Dict[str, Any]]]]] = None
) -> Dict[str, Any]:
    """
    Compute global optimality for an attack across multiple models (Stage 4).
    
    Global optimality is the average of local optimality scores across all models.
    This measures how consistently good an attack performs across different defenses.
    
    Args:
        attack_results_per_model: Dict mapping model_name -> attack_results from run_attack()
        threat_model: Threat model to analyze ('linf', 'l0', 'l1', 'l2')
        reference_per_model: Optional dict mapping model_name -> reference results
            - If None, will compute ensemble for each model from attack_results_per_model
            - Can be single attack results dict or list of attack results per model
            
    Returns:
        Dictionary with:
            - 'global_optimality': Average optimality across all models [0, 1]
            - 'local_optimality': Dict mapping model_name -> local optimality score
            - 'std_optimality': Standard deviation of local optimality
            - 'n_models': Number of models evaluated
            - 'worst_model': (model_name, score) for worst performing model
            - 'best_model': (model_name, score) for best performing model
            
    Example:
        >>> # Run attack on multiple models
        >>> results_per_model = {}
        >>> for model_name in ['Standard', 'Carmon2019Unlabeled', 'Wong2020Fast']:
        >>>     model = load_model(model_name, 'cifar10', 'Linf')
        >>>     results = run_attack(model, dataset, pgd, 'linf', device)
        >>>     results_per_model[model_name] = results
        >>>     
        >>> # Compute global optimality
        >>> global_opt = compute_global_optimality(results_per_model)
        >>> print(f"Global optimality: {global_opt['global_optimality']:.2%}")
    """
    if len(attack_results_per_model) == 0:
        raise ValueError("attack_results_per_model is empty")
    
    local_optimalities = {}
    
    for model_name, attack_results in attack_results_per_model.items():
        # Determine reference for this model
        if reference_per_model is not None and model_name in reference_per_model:
            reference = reference_per_model[model_name]
        else:
            # No reference provided, use self as reference (optimality = 1.0)
            reference = attack_results
        
        # Compute local optimality
        try:
            opt_result = compute_local_optimality(
                attack_results=attack_results,
                reference_results=reference,
                threat_model=threat_model
            )
            local_optimalities[model_name] = opt_result['optimality']
        except Exception as e:
            # Skip this model if computation fails
            print(f"Warning: Failed to compute optimality for {model_name}: {e}")
            continue
    
    if len(local_optimalities) == 0:
        raise ValueError("Could not compute optimality for any model")
    
    # Compute global statistics
    scores = list(local_optimalities.values())
    global_optimality = float(np.mean(scores))
    std_optimality = float(np.std(scores))
    
    # Find best and worst models
    worst_model = min(local_optimalities.items(), key=lambda x: x[1])
    best_model = max(local_optimalities.items(), key=lambda x: x[1])
    
    return {
        'global_optimality': global_optimality,
        'local_optimality': local_optimalities,
        'std_optimality': std_optimality,
        'n_models': len(local_optimalities),
        'worst_model': worst_model,
        'best_model': best_model
    }


def create_attack_leaderboard(
    attacks_results_per_model: Dict[str, Dict[str, Dict[str, Any]]],
    threat_model: str = 'linf',
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create attack leaderboard based on global optimality (Stage 5).
    
    Ranks attacks by their global optimality scores across multiple models.
    Higher score = better attack = closer to ensemble lower bound.
    
    Args:
        attacks_results_per_model: Nested dict:
            attack_name -> model_name -> attack_results from run_attack()
        threat_model: Threat model to analyze
        top_k: Optional, return only top K attacks
        
    Returns:
        Dictionary with:
            - 'leaderboard': List of (attack_name, global_optimality) tuples, sorted descending
            - 'attack_details': Dict mapping attack_name -> full global_optimality result
            - 'ensemble_per_model': Dict mapping model_name -> ensemble distances
            - 'n_attacks': Number of attacks evaluated
            - 'n_models': Number of models evaluated
            
    Example:
        >>> # Run multiple attacks on multiple models
        >>> attacks_results = {}
        >>> for attack_name, attack_fn in [('PGD', pgd), ('APGD', apgd), ('DeepFool', deepfool)]:
        >>>     attacks_results[attack_name] = {}
        >>>     for model_name in ['Standard', 'Carmon2019Unlabeled']:
        >>>         model = load_model(model_name, 'cifar10', 'Linf')
        >>>         results = run_attack(model, dataset, attack_fn, 'linf', device)
        >>>         attacks_results[attack_name][model_name] = results
        >>>         
        >>> # Create leaderboard
        >>> leaderboard = create_attack_leaderboard(attacks_results, top_k=5)
        >>> print("Attack Leaderboard:")
        >>> for rank, (name, score) in enumerate(leaderboard['leaderboard'], 1):
        >>>     print(f"{rank}. {name}: {score:.2%}")
    """
    if len(attacks_results_per_model) == 0:
        raise ValueError("attacks_results_per_model is empty")
    
    # Get model names (assume all attacks tested on same models)
    first_attack = next(iter(attacks_results_per_model.values()))
    model_names = list(first_attack.keys())
    
    if len(model_names) == 0:
        raise ValueError("No models found in results")
    
    # Step 1: Compute ensemble (best distances) for each model
    # Ensemble = element-wise minimum across all attacks
    ensemble_per_model = {}
    
    for model_name in model_names:
        # Collect all attacks' distances for this model
        distances_list = []
        for attack_name, results_per_model in attacks_results_per_model.items():
            if model_name in results_per_model:
                distances = np.array(results_per_model[model_name].get('distances', {}).get(threat_model, []))
                if len(distances) > 0:
                    distances_list.append(distances)
        
        if len(distances_list) == 0:
            raise ValueError(f"No valid distances for model {model_name}")
        
        # Compute ensemble for this model
        ensemble = distances_list[0].copy()
        for dist in distances_list[1:]:
            ensemble = ensemble_distances(ensemble, dist)
        
        ensemble_per_model[model_name] = ensemble
    
    # Step 2: Compute global optimality for each attack
    attack_global_optimalities = {}
    
    for attack_name, results_per_model in attacks_results_per_model.items():
        # Create reference dict using ensemble for each model
        reference_per_model = {}
        for model_name in model_names:
            if model_name in results_per_model:
                # Create a fake attack_results dict with ensemble distances
                reference_per_model[model_name] = {
                    'distances': {threat_model: ensemble_per_model[model_name].tolist()},
                    'adv_success': [1] * len(ensemble_per_model[model_name]),
                    'ori_success': [1] * len(ensemble_per_model[model_name])
                }
        
        try:
            global_opt = compute_global_optimality(
                attack_results_per_model=results_per_model,
                threat_model=threat_model,
                reference_per_model=reference_per_model
            )
            attack_global_optimalities[attack_name] = global_opt
        except Exception as e:
            print(f"Warning: Failed to compute global optimality for {attack_name}: {e}")
            continue
    
    if len(attack_global_optimalities) == 0:
        raise ValueError("Could not compute global optimality for any attack")
    
    # Step 3: Create leaderboard (sorted by global optimality)
    leaderboard = sorted(
        [(name, result['global_optimality']) for name, result in attack_global_optimalities.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Apply top_k filter if specified
    if top_k is not None and top_k > 0:
        leaderboard = leaderboard[:top_k]
    
    return {
        'leaderboard': leaderboard,
        'attack_details': attack_global_optimalities,
        'ensemble_per_model': ensemble_per_model,
        'n_attacks': len(attack_global_optimalities),
        'n_models': len(model_names),
        'models': model_names
    }


def format_leaderboard(
    leaderboard_result: Dict[str, Any],
    include_details: bool = True,
    max_decimals: int = 4
) -> str:
    """
    Format leaderboard results as a readable string.
    
    Args:
        leaderboard_result: Output from create_attack_leaderboard()
        include_details: Include per-model breakdown for each attack
        max_decimals: Number of decimal places for scores
        
    Returns:
        Formatted string representation of the leaderboard
        
    Example:
        >>> leaderboard = create_attack_leaderboard(attacks_results)
        >>> print(format_leaderboard(leaderboard))
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"ATTACK LEADERBOARD - Global Optimality Ranking")
    lines.append(f"Models: {leaderboard_result['n_models']}, Attacks: {leaderboard_result['n_attacks']}")
    lines.append("=" * 80)
    lines.append("")
    
    # Main leaderboard
    lines.append(f"{'Rank':<6} {'Attack':<30} {'Global Opt':<12} {'Std Dev':<10}")
    lines.append("-" * 80)
    
    for rank, (attack_name, global_score) in enumerate(leaderboard_result['leaderboard'], 1):
        details = leaderboard_result['attack_details'].get(attack_name, {})
        std = details.get('std_optimality', 0.0)
        lines.append(f"{rank:<6} {attack_name:<30} {global_score:<12.{max_decimals}f} {std:<10.{max_decimals}f}")
    
    # Detailed breakdown if requested
    if include_details:
        lines.append("")
        lines.append("=" * 80)
        lines.append("PER-MODEL BREAKDOWN")
        lines.append("=" * 80)
        
        for attack_name, _ in leaderboard_result['leaderboard']:
            details = leaderboard_result['attack_details'].get(attack_name, {})
            local_opts = details.get('local_optimality', {})
            
            lines.append("")
            lines.append(f"{attack_name}:")
            lines.append(f"  Global: {details.get('global_optimality', 0):.{max_decimals}f}")
            
            # Sort models by local optimality
            sorted_models = sorted(local_opts.items(), key=lambda x: x[1], reverse=True)
            for model_name, local_score in sorted_models:
                lines.append(f"    {model_name:<30} {local_score:.{max_decimals}f}")
    
    return "\n".join(lines)


def compare_attacks_global(
    attack_results_list: List[Dict[str, Dict[str, Any]]],
    attack_names: List[str],
    threat_model: str = 'linf'
) -> Dict[str, Any]:
    """
    Simplified API: compare attacks' global optimality given list of results per attack.
    
    Args:
        attack_results_list: List where each element is Dict[model_name -> attack_results]
        attack_names: Names for the attacks
        threat_model: Threat model to analyze
        
    Returns:
        Same as create_attack_leaderboard()
        
    Example:
        >>> # Collect results
        >>> pgd_results = {model_name: run_attack(...) for model_name in models}
        >>> apgd_results = {model_name: run_attack(...) for model_name in models}
        >>> 
        >>> # Compare
        >>> comparison = compare_attacks_global(
        >>>     [pgd_results, apgd_results],
        >>>     ['PGD', 'APGD']
        >>> )
    """
    if len(attack_results_list) != len(attack_names):
        raise ValueError("attack_results_list and attack_names must have same length")
    
    # Convert to format expected by create_attack_leaderboard
    attacks_results_per_model = {}
    for attack_name, results_per_model in zip(attack_names, attack_results_list):
        attacks_results_per_model[attack_name] = results_per_model
    
    return create_attack_leaderboard(
        attacks_results_per_model=attacks_results_per_model,
        threat_model=threat_model
    )
