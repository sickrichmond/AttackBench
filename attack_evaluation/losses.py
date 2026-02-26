"""
Loss functions for adversarial attacks.

This module provides loss functions commonly used in adversarial attacks,
replacing dependencies from adv_lib.utils.losses.
"""

import torch
from torch import Tensor


def difference_of_logits(logits: Tensor, labels: Tensor, targeted: bool = False) -> Tensor:
    """
    Compute the Difference of Logits (DL) loss.
    
    DL loss is defined as:
    - For untargeted: logit[true_class] - max(logit[other_classes])
    - For targeted: max(logit[other_classes]) - logit[target_class]
    
    Positive values indicate misclassification (successful attack).
    
    Args:
        logits: Model output logits, shape (batch_size, num_classes)
        labels: True labels (untargeted) or target labels (targeted), shape (batch_size,)
        targeted: Whether this is a targeted attack
        
    Returns:
        DL loss per sample, shape (batch_size,)
    """
    batch_size, num_classes = logits.shape
    
    # Get logit values for the target/true class
    target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    
    # Create a mask for other classes
    mask = torch.ones_like(logits).scatter_(1, labels.unsqueeze(1), 0.0)
    
    # Get maximum logit among other classes
    other_logits = (logits * mask + (1 - mask) * float('-inf')).max(dim=1)[0]
    
    if targeted:
        # For targeted: want other_logits > target_logits (positive when successful)
        return other_logits - target_logits
    else:
        # For untargeted: want target_logits < other_logits (positive when successful)
        return target_logits - other_logits
