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

from .distances import l0_distances, l1_distances, l2_distances, linf_distances

# Import RobustBench components
from robustbench import load_model as rb_load_model
from robustbench.loaders import make_custom_dataset

from .attacks.ingredient import get_attack
from .datasets.ingredient import get_loader
from .models.ingredient import get_model
from .utils import run_attack as _run_attack_impl, set_seed

# Import get_stats from metrics package (no duplication!)
from .metrics.analysis import get_stats

# W&B integration for cached distances
from attackbench.wandb_utils import get_precompiled_distances

# Supported metrics
METRICS = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


def _extract_metadata(model, dataset, attack):
    """
    Extract metadata from objects for W&B caching.
    
    Returns:
        tuple: (dataset_name, model_name, attack_name, attack_lib)
               Any value may be None if extraction fails.
    """
    # Extract dataset name from DataLoader
    dataset_name = getattr(dataset, '_attackbench_dataset', None)
    
    # Extract model name from model
    model_name = getattr(model, '_attackbench_model', None)
    # Fallback: try model.model for wrapped models
    if model_name is None and hasattr(model, 'model'):
        model_name = getattr(model.model, '_attackbench_model', None)
    
    # Extract attack name and lib from attack callable
    attack_name = getattr(attack, '_attackbench_name', None)
    attack_lib = getattr(attack, '_attackbench_lib', None)
    
    # Fallback: try to get name from partial/function
    if attack_name is None:
        if hasattr(attack, 'func'):  # functools.partial
            attack_name = getattr(attack.func, '__name__', None)
        else:
            attack_name = getattr(attack, '__name__', None)
    
    # Fallback: try to infer library from module
    if attack_lib is None:
        module = None
        if hasattr(attack, 'func'):
            module = getattr(attack.func, '__module__', '')
        else:
            module = getattr(attack, '__module__', '')
        
        if module:
            # Extract library name from module path
            # e.g., 'attack_evaluation.attacks.foolbox.wrapper' -> 'foolbox'
            parts = module.split('.')
            for lib_name in ['foolbox', 'torchattacks', 'adv_lib', 'art', 'cleverhans', 'deeprobust', 'original']:
                if lib_name in parts:
                    attack_lib = lib_name
                    break
    
    return dataset_name, model_name, attack_name, attack_lib


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
    include_metadata: bool = False,  # NEW: Control extra data
    # Metadata for W&B integration and optimality computation
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    attack_lib: Optional[str] = None,
    # Caching options
    use_cached: bool = True,
    cache_dir: str = "./cache",
    **kwargs
) -> Dict[str, Any]:
    """
    Execute an adversarial attack and return ONLY ESSENTIAL RAW results.
    
    This function is MINIMAL - returns only distances and success flags.
    For ALL analysis and statistics, use attackbench.get_stats() afterwards.
    
    **Automatic Metadata Extraction:**
    When using get_loader(), get_model(), and get_attack() helpers, metadata
    (dataset_name, model_name, attack_name, attack_lib) is automatically extracted
    from the objects. You don't need to specify these parameters manually.
    
    If use_cached=True and all metadata is available, the function will first
    check W&B for existing precompiled distances with the same parameters.
    If found, returns cached results without running the attack.
    
    Args:
        model: PyTorch model to attack
        dataset: DataLoader with inputs to attack
        attack: Attack function/callable
        threat_model: Threat model ('linf', 'l2', 'l1', 'l0')
        device: Device to run on (default: auto-detect)
        save_results: Save results to disk
        save_adversarial: Save adversarial examples
        output_dir: Directory for saved results
        seed: Random seed
        debug: Debug mode
        include_metadata: Include extra metadata (predictions, queries, times, hashes)
        dataset_name: Override auto-detected dataset name
        model_name: Override auto-detected model name
        attack_name: Override auto-detected attack name
        attack_lib: Override auto-detected attack library
        use_cached: If True, check W&B for cached precompiled distances before running
        cache_dir: Directory for local cache of W&B downloads
        **kwargs: Additional arguments passed to attack
    
    Returns:
        Dict with MINIMAL RAW attack data:
        - distances: Dict[str, List[float]] (adversarial distances per norm)
        - best_optim_distances: Dict[str, List[float]] (optimal tracked distances)
        - adv_success: List[bool] (attack success per sample - needed for ASR)
        - ori_success: List[bool] (original correctness - needed for accuracy)
        
        If include_metadata=True, also includes:
        - original_predictions: List[int]
        - adversarial_predictions: List[int]
        - num_forwards: List[int]
        - num_backwards: List[int]
        - times: List[float]
        - hashes: List[str]
        - box_failures: List[bool]
        - batch_failures: List[bool]
        - targeted: bool
        
        If save_adversarial=True, also includes:
        - adv_inputs: Tensor (adversarial examples)
        - inputs: Tensor (original inputs)
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
    
    # Auto-extract metadata from objects if not provided
    auto_dataset, auto_model, auto_attack, auto_lib = _extract_metadata(model, dataset, attack)
    dataset_name = dataset_name or auto_dataset
    model_name = model_name or auto_model
    attack_name = attack_name or auto_attack
    attack_lib = attack_lib or auto_lib
    
    # Calculate n_samples from dataset for W&B lookup
    n_samples = sum(len(batch[0]) for batch in dataset)
    
    # Check for cached precompiled distances on W&B
    if use_cached and all([dataset_name, model_name, attack_name, attack_lib]):
        print(f"[AttackBench] Checking W&B for cached distances: {dataset_name}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}")
        
        cached_data = get_precompiled_distances(
            dataset=dataset_name,
            threat_model=threat_model,
            model_name=model_name,
            attack_name=attack_name,
            attack_lib=attack_lib,
            n_samples=n_samples,
            cache_dir=cache_dir
        )
        
        if cached_data is not None:
            print(f"[AttackBench] Found cached distances! Skipping attack execution.")
            
            # Return cached data with metadata
            return {
                'distances': cached_data.get('distances', {}),
                'best_optim_distances': cached_data.get('best_optim_distances', {}),
                'adv_success': cached_data.get('adv_success', []),
                'ori_success': cached_data.get('ori_success', []),
                'metadata': {
                    'dataset': dataset_name,
                    'model_name': model_name,
                    'attack_name': attack_name,
                    'attack_lib': attack_lib,
                    'threat_model': threat_model,
                    'n_samples': n_samples,
                    'source': 'wandb_cache'  # Indicate this came from cache
                }
            }
        else:
            print(f"[AttackBench] No cached distances found. Running attack...")
    
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
    
    # Return ONLY essential raw data - all statistics computed by metrics package
    n_samples = len(raw_data['adv_success'])
    
    clean_data = {
        # ALWAYS: Core attack data (needed by get_stats)
        'distances': raw_data['distances'],
        'best_optim_distances': raw_data['best_optim_distances'], 
        'adv_success': raw_data['adv_success'],
        'ori_success': raw_data['ori_success'],
        # NOTE: ASR and accuracy computed by get_stats() from success flags above
        
        # ALWAYS: Metadata for W&B integration and optimality computation
        'metadata': {
            'dataset': dataset_name,
            'model_name': model_name,
            'attack_name': attack_name,
            'attack_lib': attack_lib,
            'threat_model': threat_model,
            'n_samples': n_samples,
            'source': 'executed'  # Indicate this was freshly computed
        }
    }
    
    # OPTIONAL: Extra metadata (only if requested)
    if include_metadata:
        clean_data.update({
            'original_predictions': raw_data.get('original_predictions', []),
            'adversarial_predictions': raw_data.get('adversarial_predictions', []),
            'num_forwards': raw_data['num_forwards'],
            'num_backwards': raw_data['num_backwards'],
            'times': raw_data['times'],
            'hashes': raw_data['hashes'],
            'box_failures': raw_data['box_failures'],
            'batch_failures': raw_data['batch_failures'],
            'targeted': raw_data['targeted'],
        })
    
    # Add adversarial inputs if requested
    if save_adversarial and 'adv_inputs' in raw_data:
        clean_data['adv_inputs'] = raw_data['adv_inputs']
        clean_data['inputs'] = raw_data['inputs']
    
    # Save raw results if requested
    if save_results or save_adversarial:
        if output_dir:
            from pathlib import Path
            import json
            
            # Simple flat directory structure
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save adversarial data
            if save_adversarial and 'adv_inputs' in clean_data:
                torch.save(clean_data, save_dir / 'attack_data.pt')
            
            # Save results JSON (without large tensors)
            results_to_save = {k: v for k, v in clean_data.items() 
                             if k not in ['inputs', 'adv_inputs']}
            
            with open(save_dir / 'results.json', 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
    
    return clean_data  # SOLO DATI GREZZI - zero statistiche


# NOTE: get_stats() is now imported from metrics.analysis - no duplication!
# All analysis functions (curves, optimality, efficiency) are in the metrics package.


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

