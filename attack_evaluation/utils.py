import hashlib
import inspect  # Add this import
import random
import traceback
import warnings
from collections import defaultdict
from typing import Callable, Dict, Optional, Union

import numpy as np
import torch
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models.benchmodel_wrapper import BenchModel


def run_attack(model: BenchModel,
               loader: DataLoader,
               attack: Callable,
               targets: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               threat_model: str = 'l2',
               return_adv: bool = False,
               debug: bool = False,
               **kwargs) -> dict:  # Accepts **kwargs
    """
    Run an adversarial attack on a model using the provided data loader.
    
    Args:
        model: BenchModel wrapper around the target model
        loader: DataLoader containing input data
        attack: Attack function to execute
        targets: Target labels for targeted attacks (None for untargeted)
        metrics: Dictionary of distance metrics to compute
        threat_model: Type of threat model ('l2', 'linf', etc.)
        return_adv: Whether to return adversarial examples
        debug: Whether to run in debug mode
        **kwargs: Additional arguments passed to the attack function
        
    Returns:
        Dictionary containing attack results and statistics
    """
    
    device = next(model.parameters()).device
    targeted = True if targets is not None else False
    loader_length = len(loader)

    accuracies, ori_success, adv_success, hashes, box_failures, batch_failures = [], [], [], [], [], []
    forwards, backwards, times = [], [], []
    distances, best_optim_distances = defaultdict(list), defaultdict(list)

    if return_adv:
        all_inputs, all_adv_inputs = [], []

    for inputs, labels in tqdm(loader, ncols=80, total=loader_length):
        if return_adv:
            all_inputs.append(inputs.clone())

        # Compute hashes to ensure that input samples are identical
        for input in inputs:
            input_hash = hashlib.sha512(np.ascontiguousarray(input.numpy())).hexdigest()
            hashes.append(input_hash)

        inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
        attack_inputs, attack_labels = inputs.clone(), labels.clone()  # Ensure no in-place modification
        
        # Start tracking of the batch
        model.start_tracking(inputs=inputs, labels=labels, targeted=targeted, targets=targets,
                             tracking_metric=_default_metrics[threat_model], tracking_threat_model=threat_model)

        if debug:
            adv_inputs = _call_attack_with_kwargs(
                attack, model, attack_inputs, attack_labels, targeted, targets, **kwargs
            )
        else:
            try:
                adv_inputs = _call_attack_with_kwargs(
                    attack, model, attack_inputs, attack_labels, targeted, targets, **kwargs
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

        if return_adv:
            all_adv_inputs.append(adv_inputs.cpu().clone())

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)

        success = (adv_pred == targets) if targeted else (adv_pred != labels)
        adv_success.extend(success.cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_inputs, inputs).cpu().tolist())
            best_optim_distances[metric].extend(model.min_dist[metric].cpu().tolist())

    data = {
        'hashes': hashes,
        'targeted': targeted,
        'accuracy': sum(accuracies) / len(accuracies),
        'ori_success': ori_success,
        'adv_success': adv_success,
        'ASR': sum(adv_success) / len(adv_success),
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': dict(distances),
        'best_optim_distances': dict(best_optim_distances),
        'box_failures': box_failures,
        'batch_failures': batch_failures,
    }

    if return_adv:
        # shapes = [img.shape for img in all_inputs]
        # if len(set(shapes)) == 1:
        if len(all_inputs) > 1:
            all_inputs = torch.cat(all_inputs, dim=0)
            all_adv_inputs = torch.cat(all_adv_inputs, dim=0)
        data['inputs'] = all_inputs
        data['adv_inputs'] = all_adv_inputs

    return data


def set_seed(seed: int = None) -> None:
    """
    Set random seed for PyTorch, NumPy, and Python random module.
    
    See https://pytorch.org/docs/stable/notes/randomness.html for further details.
    
    Args:
        seed: Random seed value (None to skip seeding)
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _call_attack_with_kwargs(attack, model, inputs, labels, targeted, targets, **kwargs):
    """
    Call the attack function filtering kwargs based on its signature.
    
    This function inspects the attack's signature and only passes parameters
    that the function can accept, avoiding TypeError from unexpected kwargs.
    
    Args:
        attack: Attack function to call
        model: Model to attack
        inputs: Input tensors
        labels: True labels
        targeted: Whether attack is targeted
        targets: Target labels (for targeted attacks)
        **kwargs: Additional arguments to filter and pass
        
    Returns:
        Adversarial examples from the attack
    """
    
    # Get the attack's signature
    sig = inspect.signature(attack)
    
    # Base parameters always present
    attack_params = {
        'model': model,
        'inputs': inputs,
        'labels': labels,
    }
    
    # Add optional parameters if accepted
    if 'targeted' in sig.parameters:
        attack_params['targeted'] = targeted
    if 'targets' in sig.parameters:
        attack_params['targets'] = targets
    
    # Filter kwargs to include only those accepted
    for key, value in kwargs.items():
        if key in sig.parameters:
            attack_params[key] = value
    
    return attack(**attack_params)

