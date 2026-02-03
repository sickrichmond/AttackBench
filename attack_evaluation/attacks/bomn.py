"""
Best-of-MinNorm (BoMN) composite attack module.

For each sample: run all attacks, select the one with minimum norm.

Definition:
We define BoMN as a composite attack obtained by running a set of minimum-norm 
adversarial attacks under a fixed query budget and selecting, for each input 
sample, the perturbation with the smallest norm.
"""

import torch
import numpy as np
from typing import List, Callable, Dict, Any, Optional
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances

# Distance function mapping
DISTANCE_FUNCTIONS = {
    'l0': l0_distances,
    'l1': l1_distances,
    'l2': l2_distances,
    'linf': linf_distances,
}


def _get_attack_name(attack: Callable) -> str:
    """Extract readable name from attack callable."""
    if hasattr(attack, 'func'):  # functools.partial
        return getattr(attack.func, '__name__', 'unknown')
    return getattr(attack, '__name__', 'unknown')


def bomn_attack(
    model: nn.Module,
    dataset: DataLoader,
    attacks: List[Callable],
    threat_model: str,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Best-of-MinNorm (BoMN) composite attack.
    
    For each sample: run all attacks, select best (minimum distance).
    
    Args:
        model: PyTorch model to attack
        dataset: DataLoader with inputs to attack
        attacks: List of attack callables (e.g., [pgd, apgd, deepfool])
        threat_model: Norm to use ('l0', 'l1', 'l2', 'linf')
        device: Device to run on
        verbose: Print sample-by-sample results
        **kwargs: Additional arguments passed to attacks
        
    Returns:
        Dict with per-sample results:
        - distances: Dict[norm -> distances] - best distance for each sample
        - best_optim_distances: Same as distances
        - adv_success: List[bool] - attack success per sample
        - ori_success: List[bool] - original correctness per sample
        - best_attack_indices: List[int] - which attack won each sample
        - attack_names: List[str] - names of attacks
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if threat_model not in DISTANCE_FUNCTIONS:
        raise ValueError(f"Unknown threat model: {threat_model}. Must be one of {list(DISTANCE_FUNCTIONS.keys())}")
    
    distance_fn = DISTANCE_FUNCTIONS[threat_model]
    attack_names = [_get_attack_name(attack) for attack in attacks]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"BoMN-{threat_model.upper()}: {len(attacks)} attacks per sample")
        print(f"Attacks: {', '.join(attack_names)}")
        print(f"{'='*70}\n")
        print(f"{'Sample':<8} {'Best Attack':<20} {'Distance':<12} {'Success'}")
        print(f"{'-'*70}")
    
    # Results storage
    all_best_distances = []
    all_adv_success = []
    all_ori_success = []
    all_best_attack_indices = []
    
    # Process dataset sample by sample
    model.eval()
    sample_id = 0
    
    for batch_data in tqdm(dataset, desc="Processing", disable=not verbose):
        if isinstance(batch_data, (tuple, list)):
            inputs, labels = batch_data[0], batch_data[1]
        else:
            inputs = batch_data
            labels = None
        
        inputs = inputs.to(device)
        if labels is not None:
            labels = labels.to(device)
        
        batch_size = inputs.shape[0]
        
        # Process each sample in the batch individually
        for i in range(batch_size):
            sample_id += 1
            x = inputs[i:i+1]  # Keep batch dimension
            y = labels[i:i+1] if labels is not None else None
            
            # Check if model correctly classifies original sample
            with torch.no_grad():
                logits = model(x)
                pred = logits.argmax(dim=1)
                if y is not None:
                    originally_correct = (pred == y).item()
                else:
                    originally_correct = True
            
            all_ori_success.append(originally_correct)
            
            if not originally_correct:
                # Skip misclassified samples
                all_best_distances.append(0.0)
                all_adv_success.append(False)
                all_best_attack_indices.append(-1)
                if verbose:
                    print(f"{sample_id:<8} {'N/A':<20} {'N/A':<12} {'SKIP (misclassified)'}")
                continue
            
            # Run all attacks on this single sample
            sample_distances = []
            sample_adversarial = []
            
            for attack_idx, attack in enumerate(attacks):
                try:
                    # Call attack on single sample
                    adv_x = attack(model, x, y, **kwargs)
                    
                    # Ensure correct shape
                    if adv_x.shape != x.shape:
                        adv_x = adv_x.reshape(x.shape)
                    
                    # Compute distance
                    dist = distance_fn(x, adv_x).item()
                    
                    # Check if attack succeeded
                    with torch.no_grad():
                        adv_logits = model(adv_x)
                        adv_pred = adv_logits.argmax(dim=1)
                        if y is not None:
                            attack_succeeded = (adv_pred != y).item()
                        else:
                            attack_succeeded = (adv_pred != pred).item()
                    
                    # Only consider successful attacks
                    if attack_succeeded:
                        sample_distances.append(dist)
                        sample_adversarial.append(adv_x)
                    else:
                        sample_distances.append(float('inf'))  # Failed attack
                        sample_adversarial.append(None)
                
                except Exception as e:
                    if verbose:
                        print(f"\nWarning: {attack_names[attack_idx]} failed on sample: {e}")
                    sample_distances.append(float('inf'))
                    sample_adversarial.append(None)
            
            # Select best attack (minimum distance)
            best_idx = np.argmin(sample_distances)
            best_dist = sample_distances[best_idx]
            
            if best_dist == float('inf'):
                # All attacks failed
                all_best_distances.append(0.0)
                all_adv_success.append(False)
                all_best_attack_indices.append(-1)
                if verbose:
                    print(f"{sample_id:<8} {'NONE':<20} {0.0:<12.6f} {'FAIL (all attacks)'}")
            else:
                # At least one attack succeeded
                all_best_distances.append(best_dist)
                all_adv_success.append(True)
                all_best_attack_indices.append(best_idx)
                if verbose:
                    winner = attack_names[best_idx]
                    print(f"{sample_id:<8} {winner:<20} {best_dist:<12.6f} {'SUCCESS'}")
    
    if verbose:
        print(f"{'-'*70}\n")
    
    return {
        'distances': {threat_model: all_best_distances},
        'best_optim_distances': {threat_model: all_best_distances},
        'adv_success': all_adv_success,
        'ori_success': all_ori_success,
        'best_attack_indices': all_best_attack_indices,
        'attack_names': attack_names,
    }
