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

# Import get_stats from metrics package (no duplication!)
from .metrics.analysis import get_stats

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
    include_metadata: bool = False,  # NEW: Control extra data
    **kwargs
) -> Dict[str, Any]:
    """
    Execute an adversarial attack and return ONLY ESSENTIAL RAW results.
    
    This function is MINIMAL - returns only distances and success flags.
    For ALL analysis and statistics, use attackbench.get_stats() afterwards.
    
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
    clean_data = {
        # ALWAYS: Core attack data (needed by get_stats)
        'distances': raw_data['distances'],
        'best_optim_distances': raw_data['best_optim_distances'], 
        'adv_success': raw_data['adv_success'],
        'ori_success': raw_data['ori_success'],
        # NOTE: ASR and accuracy computed by get_stats() from success flags above
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

