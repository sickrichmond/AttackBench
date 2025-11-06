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
    Execute an adversarial attack and return basic results.
    SIMPLIFIED VERSION - no complex analysis here.
    
    Returns:
        Dict with basic attack results (ASR, accuracy, distances, timing)
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
    
    # Run attack - simplified (ONLY basic results)
    attack_data = _run_attack_impl(
        model=model,
        loader=dataset,
        attack=attack,
        metrics=METRICS,
        threat_model=threat_model,
        return_adv=save_adversarial,
        debug=debug,
        **kwargs
    )
    
    # Save results if requested
    if save_results or save_adversarial:
        _save_results(attack_data, model, dataset, attack, threat_model, 
                     output_dir, save_adversarial)
    
    return attack_data


def _save_results(attack_data, model, dataset, attack, threat_model, output_dir, save_adversarial):
    """Save results to files - simplified version"""
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
    certified_thresholds: Optional[List[float]] = None,
    save_precompiled_distances: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute advanced statistics and analysis on attack results.
    SEPARATED from run_attack for modularity.
    
    Args:
        attack_results: Results from run_attack()
        threat_model: Threat model used
        certified_thresholds: Thresholds for robust accuracy
        save_precompiled_distances: Whether to save distances
        output_dir: Output directory for saved files
        
    Returns:
        Dict with advanced statistics (optimality, curves, etc.)
    """
    
    analysis = {}
    
    # 1. Basic distance statistics
    distances = attack_results.get('distances', {})
    
    for norm, dist_values in distances.items():
        dist_array = np.array(dist_values)
        
        # Descriptive statistics
        analysis[f'{norm}_mean_distance'] = float(np.mean(dist_array))
        analysis[f'{norm}_median_distance'] = float(np.median(dist_array))
        analysis[f'{norm}_std_distance'] = float(np.std(dist_array))
        analysis[f'{norm}_min_distance'] = float(np.min(dist_array))
        analysis[f'{norm}_max_distance'] = float(np.max(dist_array))
        
        # Percentiles
        analysis[f'{norm}_p25_distance'] = float(np.percentile(dist_array, 25))
        analysis[f'{norm}_p75_distance'] = float(np.percentile(dist_array, 75))
        analysis[f'{norm}_p95_distance'] = float(np.percentile(dist_array, 95))
        
        # Distances only for successful adversarial examples
        adv_success = np.array(attack_results.get('adv_success', []))
        if adv_success.any():
            successful_distances = dist_array[adv_success]
            if len(successful_distances) > 0:
                analysis[f'{norm}_successful_mean_distance'] = float(np.mean(successful_distances))
                analysis[f'{norm}_successful_median_distance'] = float(np.median(successful_distances))
    
    # 2. Optimality computation
    if threat_model in distances and threat_model in attack_results.get('best_optim_distances', {}):
        main_distances = np.array(distances[threat_model])
        best_distances = np.array(attack_results['best_optim_distances'][threat_model])
        
        try:
            from analysis.utils import eval_optimality
            optimality = eval_optimality(main_distances, best_distances)
            analysis['optimality'] = float(optimality)
        except ImportError:
            print("Warning: Cannot import eval_optimality from analysis.utils")
            analysis['optimality'] = None
    
    # 3. Security evaluation curves
    if threat_model in distances:
        distances_array = np.array(distances[threat_model])
        success_mask = np.array(attack_results.get('adv_success', []))
        
        # Robust accuracy curve
        curve_data = _compute_robust_accuracy_curve_fallback(distances_array, success_mask)
        analysis['robust_accuracy_curve'] = curve_data
        
        # Certified robustness metrics
        if certified_thresholds is None:
            if threat_model == 'linf':
                certified_thresholds = [4/255, 8/255, 16/255]
            elif threat_model == 'l2':
                certified_thresholds = [0.5, 1.0, 2.0]
            else:
                certified_thresholds = []
        
        if certified_thresholds:
            cert_metrics = _compute_certified_robustness_metrics(
                distances_array, success_mask, certified_thresholds
            )
            analysis.update(cert_metrics)
    
    # 4. Precompiled distances
    if save_precompiled_distances and output_dir:
        precompiled_path = _save_precompiled_distances_fallback(attack_results, threat_model, output_dir)
        analysis['precompiled_distances_path'] = precompiled_path
    
    return analysis


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

