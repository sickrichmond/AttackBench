import hashlib
import inspect
import random
import traceback
import warnings
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
from typing import Dict, Any, Optional, Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# W&B integration for cached distances
from .wandb.utils import get_precompiled_distances


def _set_seed(seed: int = None) -> None:
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _call_attack(attack, model, inputs, labels, targeted, targets, **kwargs):
    """
    Call the attack function, filtering kwargs to match its signature.
    """
    sig = inspect.signature(attack)

    attack_params = {
        'model': model,
        'inputs': inputs,
        'labels': labels,
    }

    if 'targeted' in sig.parameters:
        attack_params['targeted'] = targeted
    if 'targets' in sig.parameters:
        attack_params['targets'] = targets

    for key, value in kwargs.items():
        if key in sig.parameters:
            attack_params[key] = value

    return attack(**attack_params)


def _extract_metadata(model, dataset, attack):
    """
    Extract metadata from objects for W&B caching.
    
    Returns:
        tuple: (dataset_name, model_name, attack_name, attack_lib)
               Any value may be None if extraction fails.
    """
    # Extract dataset name from DataLoader (set by get_loader)
    dataset_name = getattr(dataset, '_attackbench_dataset', None)

    # Extract model name from model (set by attackbench.load_model / get_model)
    model_name = getattr(model, '_attackbench_model', None)
    # Fallback: try model.model for BenchModel-wrapped models
    if model_name is None and hasattr(model, 'model'):
        model_name = getattr(model.model, '_attackbench_model', None)

    # Fallback: read dataset from model if loader didn't provide it
    # (set by attackbench.load_model which receives dataset as parameter)
    if dataset_name is None:
        dataset_name = getattr(model, '_attackbench_dataset', None)
        if dataset_name is None and hasattr(model, 'model'):
            dataset_name = getattr(model.model, '_attackbench_dataset', None)
    
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
    _set_seed(seed)

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
        key = f"{dataset_name}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}".lower()
        print(f"[AttackBench] Checking W&B for cached distances: {key}")

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
            return {
                'distances': cached_data.get('distances', {}),
                'best_optim_distances': cached_data.get('best_optim_distances', {}),
                'adv_success': cached_data.get('adv_success', []),
                'ori_success': cached_data.get('ori_success', []),
                'hashes': cached_data.get('hashes', []),
                'metadata': {
                    'dataset': dataset_name,
                    'model_name': model_name,
                    'attack_name': attack_name,
                    'attack_lib': attack_lib,
                    'threat_model': threat_model,
                    'n_samples': n_samples,
                    'source': 'wandb_cache'
                }
            }
        else:
            print(f"[AttackBench] No cached distances found. Running attack...")

    # ── Execute attack (batch loop) ──────────────────────────────────────
    from .adv_lib_sub import _default_metrics
    metrics = _default_metrics

    targeted = False
    loader_length = len(dataset)

    accuracies, ori_success, adv_success = [], [], []
    hashes_list, box_failures, batch_failures = [], [], []
    predictions, adv_predictions = [], []
    forwards, backwards, times = [], [], []
    distances, best_optim_distances = defaultdict(list), defaultdict(list)

    if save_adversarial:
        all_inputs, all_adv_inputs = [], []

    for inputs, labels in tqdm(dataset, ncols=80, total=loader_length):
        if save_adversarial:
            all_inputs.append(inputs.clone())

        # Compute hashes to ensure input samples are identical
        for inp in inputs:
            input_hash = hashlib.sha512(np.ascontiguousarray(inp.numpy())).hexdigest()
            hashes_list.append(input_hash)

        inputs, labels = inputs.to(device), labels.to(device)
        attack_inputs, attack_labels = inputs.clone(), labels.clone()

        # Start tracking of the batch
        model.start_tracking(
            inputs=inputs, labels=labels, targeted=targeted, targets=None,
            tracking_metric=metrics[threat_model], tracking_threat_model=threat_model
        )

        if debug:
            adv_inputs = _call_attack(
                attack, model, attack_inputs, attack_labels, targeted, None,
                threat_model=threat_model, **kwargs
            )
        else:
            try:
                adv_inputs = _call_attack(
                    attack, model, attack_inputs, attack_labels, targeted, None,
                    threat_model=threat_model, **kwargs
                )
                batch_failures.append(False)
            except Exception:
                warnings.warn(f'Error running batch for {attack}')
                traceback.print_exc()
                batch_failures.append(True)
                adv_inputs = inputs

        model.end_tracking()
        adv_inputs.detach_()
        times.append(model.elapsed_time)
        forwards.extend(model.num_forwards.cpu().tolist())
        backwards.extend(model.num_backwards.cpu().tolist())

        # Original inputs
        accuracies.extend(model.correct.cpu().tolist())
        ori_success.extend(model.ori_success.cpu().tolist())

        # Checking box constraint
        batch_box_failures = ((adv_inputs < 0) | (adv_inputs > 1)).flatten(1).any(1)
        box_failures.extend(batch_box_failures.cpu().tolist())

        if batch_box_failures.any():
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_inputs.clamp_(min=0, max=1)

        if save_adversarial:
            all_adv_inputs.append(adv_inputs.cpu().clone())

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)

        ori_logits = model(inputs)
        ori_pred = ori_logits.argmax(dim=1)
        predictions.extend(ori_pred.cpu().tolist())
        adv_predictions.extend(adv_pred.cpu().tolist())

        success = (adv_pred != labels)
        adv_success.extend(success.cpu().tolist())

        for metric_name, metric_func in metrics.items():
            distances[metric_name].extend(metric_func(adv_inputs, inputs).cpu().tolist())
            best_optim_distances[metric_name].extend(model.min_dist[metric_name].cpu().tolist())

    # ── Package results ──────────────────────────────────────────────────
    n_samples = len(adv_success)
    
    clean_data = {
        'distances': dict(distances),
        'best_optim_distances': dict(best_optim_distances),
        'adv_success': adv_success,
        'ori_success': ori_success,
        'hashes': hashes_list,  # Always included for sample identity tracking
        'metadata': {
            'dataset': dataset_name,
            'model_name': model_name,
            'attack_name': attack_name,
            'attack_lib': attack_lib,
            'threat_model': threat_model,
            'n_samples': n_samples,
            'source': 'executed'
        }
    }

    # OPTIONAL: Extra metadata (only if requested)
    if include_metadata:
        clean_data.update({
            'original_predictions': predictions,
            'adversarial_predictions': adv_predictions,
            'num_forwards': forwards,
            'num_backwards': backwards,
            'times': times,
            'box_failures': box_failures,
            'batch_failures': batch_failures,
            'targeted': targeted,
        })
    
    # Add adversarial inputs if requested
    if save_adversarial:
        if len(all_inputs) > 1:
            all_inputs = torch.cat(all_inputs, dim=0)
            all_adv_inputs = torch.cat(all_adv_inputs, dim=0)
        clean_data['adv_inputs'] = all_adv_inputs
        clean_data['inputs'] = all_inputs
    
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
    
    return clean_data

