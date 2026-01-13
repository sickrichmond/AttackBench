import argparse
import json
import logging
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import torch
from typing import Union, Dict, Any, Optional, Callable, List
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances

# Import RobustBench components
from robustbench import load_model as rb_load_model
from robustbench.loaders import make_custom_dataset

from .attacks.ingredient import get_attack
from .datasets.ingredient import get_loader
from .models.ingredient import get_model
from .utils import run_attack as _run_attack_impl, set_seed

# Supported metrics
METRICS = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


def run_attack(
    model: nn.Module,
    dataset: DataLoader,
    attack: Callable,
    threat_model: str,
    device: Optional[torch.device] = None,
    save_results: bool = False,
    save_adversarial: bool = False,
    output_dir: Optional[str] = None,
    seed: int = 42,
    debug: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Execute an adversarial attack and return ONLY RAW results.
    
    This function is now MINIMAL - NO statistics computed here.
    For ALL analysis, use attackbench.get_stats() afterwards.
    
    Returns:
        Dict with ONLY RAW attack data:
        - distances: Dict[str, List[float]] (per-sample distances for each norm)
        - best_optim_distances: Dict[str, List[float]] (optimal distances if tracked)
        - adv_success: List[bool] (per-sample attack success)
        - ori_success: List[bool] (per-sample original success)
        - original_predictions: List[int] (original model predictions)  
        - adversarial_predictions: List[int] (adversarial predictions)
        - num_forwards: List[int] (forward queries per sample)
        - num_backwards: List[int] (backward queries per sample)
        - times: List[float] (time per batch)
        - hashes: List[str] (sample identification)
        - box_failures: List[bool] (constraint violations)
        - batch_failures: List[bool] (batch processing failures)
        - targeted: bool (attack type)
        - adv_inputs: Optional[Tensor] (if save_adversarial=True)
    """
    
    # Input validation
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)}")
    
    if not isinstance(dataset, DataLoader):
        raise TypeError(f"dataset must be DataLoader, got {type(dataset)}")
    
    if not callable(attack):
        raise TypeError(f"attack must be callable, got {type(attack)}")
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed
    set_seed(seed)
    
    # Wrap model if necessary
    if not hasattr(model, 'start_tracking'):
        from .models.benchmodel_wrapper import BenchModel
        model = BenchModel(model)
    
    model.to(device)
    
    # Add norm to kwargs for custom attacks
    kwargs['norm'] = threat_model
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty - no inputs to attack")
    
    # Run attack - SOLO dati grezzi
    raw_data = _run_attack_impl(
        model=model,
        loader=dataset,
        attack=attack,
        metrics=METRICS,  # Solo calcolo distanze base
        threat_model=threat_model,
        return_adv=save_adversarial,
        debug=debug,
        **kwargs
    )
    
    # Return ONLY raw data - all statistics computed by metrics package
    clean_data = {
        # Raw attack data
        'distances': raw_data['distances'],
        'best_optim_distances': raw_data['best_optim_distances'], 
        'adv_success': raw_data['adv_success'],
        'ori_success': raw_data['ori_success'],
        'num_forwards': raw_data['num_forwards'],
        'num_backwards': raw_data['num_backwards'],
        'times': raw_data['times'],
        'hashes': raw_data['hashes'],
        'box_failures': raw_data['box_failures'],
        'batch_failures': raw_data['batch_failures'],
        'targeted': raw_data['targeted'],
        'original_predictions': raw_data.get('original_predictions', []),
        'adversarial_predictions': raw_data.get('adversarial_predictions', []),
        # NOTE: ASR and accuracy computed by get_stats() in metrics package
    }
    
    # Add adversarial inputs if requested
    if save_adversarial and 'adv_inputs' in raw_data:
        clean_data['adv_inputs'] = raw_data['adv_inputs']
        clean_data['inputs'] = raw_data['inputs']
    
    # Save raw results if requested
    if save_results or save_adversarial:
        _save_raw_results(clean_data, output_dir, save_adversarial)
    
    return clean_data  # SOLO DATI GREZZI - zero statistiche


def _save_raw_results(attack_data, output_dir, save_adversarial):
    """Save raw results to files - semplificata"""
    if output_dir is None:
        return
        
    # Generic names for objects
    model_name = getattr(model, '__class__', type(model)).__name__
    dataset_name = 'custom_dataset'
    attack_name = getattr(attack, '__name__', 'custom_attack')
    
    subdirs = [dataset_name, threat_model, model_name, attack_name]
    save_dir = Path(output_dir).joinpath(*subdirs)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save adversarial data
    if save_adversarial:
        torch.save(attack_data, save_dir / 'attack_data.pt')
    
    # Save results JSON (without large tensors)
    results = attack_data.copy()
    if 'inputs' in results:
        del results['inputs']
    if 'adv_inputs' in results:
        del results['adv_inputs']
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

def get_stats(
    attack_results: Dict[str, Any], 
    threat_model: str,
    # NEW PARAMETERS for granular control
    include_optimality: bool = True,
    include_curves: bool = True, 
    include_certified: bool = True,
    certified_thresholds: Optional[List[float]] = None,
    include_efficiency: bool = False,
    save_precompiled: bool = False,
    output_dir: Optional[str] = None,
    **analysis_kwargs
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics from RAW attack results.
    
    NOW COMPUTES ALL STATISTICS including basic ones like ASR and accuracy.
    
    Args:
        attack_results: RAW results from run_attack()
        threat_model: Threat model used
        include_optimality: Compute optimality scores
        include_curves: Compute robust accuracy curves
        include_certified: Compute certified robustness metrics
        include_efficiency: Compute query efficiency metrics
        save_precompiled: Save precompiled distances
        **analysis_kwargs: Additional arguments for specific analyses
        
    Returns:
        Dict with comprehensive analysis results
    """
    
    stats = {}
    
    # 0. BASIC METRICS (now computed here instead of run_attack)
    adv_success = attack_results.get('adv_success', [])
    ori_success = attack_results.get('ori_success', [])
    
    if adv_success:
        stats['ASR'] = sum(adv_success) / len(adv_success)  # Attack Success Rate
        
    if ori_success:
        stats['accuracy'] = sum(ori_success) / len(ori_success)  # Model accuracy
    
    # 1. SEMPRE: Statistiche delle distanze  
    distances = attack_results.get('distances', {})
    for norm, dist_values in distances.items():
        if dist_values:
            from attack_evaluation.metrics.distances import compute_distance_statistics
            dist_stats = compute_distance_statistics(dist_values)
            # Aggiungi prefisso norm a tutte le chiavi
            for key, value in dist_stats.items():
                stats[f'{norm}_{key}'] = value
    
    # 2. OPZIONALE: Calcolo ottimalità  
    if include_optimality and threat_model in distances:
        best_distances = attack_results.get('best_optim_distances', {}).get(threat_model, [])
        if best_distances:
            from attack_evaluation.metrics.distances import compute_optimality_score
            optimality = compute_optimality_score(distances[threat_model], best_distances)
            stats['optimality'] = optimality
    
    # 3. OPZIONALE: Curve di robust accuracy
    if include_curves and threat_model in distances:
        distances_array = np.array(distances[threat_model])
        success_mask = np.array(adv_success)
        
        from attack_evaluation.metrics.curves import compute_robust_accuracy_curve
        curve_data = compute_robust_accuracy_curve(distances_array, success_mask)
        stats['robust_accuracy_curve'] = curve_data
        
        # AUC della curva
        if curve_data['thresholds']:
            from attack_evaluation.metrics.curves import compute_auc_robust_accuracy
            auc = compute_auc_robust_accuracy(curve_data['thresholds'], curve_data['robust_accuracies'])
            stats['robust_accuracy_auc'] = auc
    
    # 4. OPZIONALE: Metriche certificate
    if include_certified and threat_model in distances:
        distances_array = np.array(distances[threat_model])
        success_mask = np.array(adv_success)
        
        from attack_evaluation.metrics.curves import compute_certified_robustness_metrics
        cert_metrics = compute_certified_robustness_metrics(
            distances_array, success_mask, threat_model, certified_thresholds
        )
        stats.update(cert_metrics)
    
    # 5. OPZIONALE: Efficienza query
    if include_efficiency:
        num_forwards = attack_results.get('num_forwards', [])
        if num_forwards and threat_model in distances:
            from attack_evaluation.metrics.distances import compute_attack_efficiency
            efficiency_metrics = compute_attack_efficiency(
                distances[threat_model], num_forwards
            )
            stats.update(efficiency_metrics)
    
    # 6. OPZIONALE: Salvataggio precompilato
    if save_precompiled and output_dir:
        from attack_evaluation.metrics.storage import save_precompiled_distances
        saved_path = save_precompiled_distances(attack_results, threat_model, output_dir)
        stats['precompiled_path'] = saved_path
    
    return stats


def _compute_robust_accuracy_curve_fallback(distances: np.ndarray, success_mask: np.ndarray) -> Dict[str, List[float]]:
    """Fallback implementation for robust accuracy curves"""
    # Sort distances
    sorted_distances = np.sort(distances)
    unique_distances = np.unique(sorted_distances)
    
    # Take a reasonable subset of thresholds
    n_thresholds = min(100, len(unique_distances))
    indices = np.linspace(0, len(unique_distances)-1, n_thresholds, dtype=int)
    thresholds = unique_distances[indices]
    
    robust_accuracies = []
    
    for threshold in thresholds:
        # An example is "robust" at this threshold if:
        # 1. The attack was not successful, OR
        # 2. The attack was successful but with distance > threshold
        robust_mask = (~success_mask) | (distances > threshold)
        robust_accuracy = robust_mask.mean()
        robust_accuracies.append(float(robust_accuracy))
    
    return {
        'thresholds': thresholds.tolist(),
        'robust_accuracies': robust_accuracies
    }


def _compute_certified_robustness_metrics(
    distances: np.ndarray,
    success_mask: np.ndarray,
    thresholds: List[float]
) -> Dict[str, float]:
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


def _save_precompiled_distances_fallback(attack_data: Dict[str, Any], threat_model: str, output_dir: str) -> str:
    """Fallback implementation for saving precompiled distances"""
    from pathlib import Path
    import json
    
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    precompiled_data = {
        'threat_model': threat_model,
        'attack_metadata': {
            'ASR': attack_data.get('ASR'),
            'accuracy': attack_data.get('accuracy'),
            'total_samples': len(attack_data.get('adv_success', [])),
            'successful_attacks': sum(attack_data.get('adv_success', [])),
        },
        'distances': attack_data.get('distances', {}),
        'success_masks': {
            'adv_success': attack_data.get('adv_success', []),
            'ori_success': attack_data.get('ori_success', []),
        },
        'sample_hashes': attack_data.get('hashes', []),
        'timing_data': {
            'total_time': sum(attack_data.get('times', [])),
            'avg_time_per_sample': np.mean(attack_data.get('times', [])) if attack_data.get('times') else 0,
            'num_forwards': attack_data.get('num_forwards', []),
            'num_backwards': attack_data.get('num_backwards', []),
        }
    }
    
    # Save in compressed NumPy format
    distances_file = save_dir / 'precompiled_distances.npz'
    
    # Convert everything to numpy arrays for saving
    save_data = {}
    for key, value in precompiled_data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                save_data[f'{key}_{subkey}'] = np.array(subvalue)
        else:
            save_data[key] = np.array(value) if isinstance(value, list) else value
    
    np.savez_compressed(distances_file, **save_data)
    
    # Also save metadata in JSON for readability
    metadata_file = save_dir / 'precompiled_metadata.json'
    metadata = {
        'threat_model': precompiled_data['threat_model'],
        'attack_metadata': precompiled_data['attack_metadata'],
        'available_norms': list(precompiled_data['distances'].keys()),
        'file_path': str(distances_file),
        'total_samples': precompiled_data['attack_metadata']['total_samples']
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return str(distances_file)


# These functions are no longer necessary - removed:
# def _load_model(model, dataset_hint='cifar10'): 
# def _load_dataset(dataset, batch_size):  
# def _load_attack(attack, threat_model): 


# Maintain CLI compatibility (OPTIONAL - can be removed)
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Attack Evaluation Script')
    
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--attack', type=str, required=True)
    parser.add_argument('--threat_model', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--save_adv', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    
    return parser.parse_args()

def main():
    """CLI entry point (OPTIONAL)"""
    args = parse_arguments()
    
    results = run_attack(
        model=args.model,
        dataset=args.dataset,
        attack=args.attack,
        threat_model=args.threat_model,
        batch_size=args.batch_size,
        save_results=True,
        save_adversarial=args.save_adv,
        output_dir=args.output_dir,
        seed=args.seed,
        debug=args.debug
    )
    
    print(f"Attack completed! ASR: {results.get('ASR', 'N/A'):.2%}")

if __name__ == '__main__':
    main()

