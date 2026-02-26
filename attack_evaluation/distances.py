"""
Distance metrics for measuring perturbation norms.

These implementations replace the adv_lib.distances.lp_norms module
to remove external dependency while maintaining the same functionality.

When 'dim' is specified, it means "reduce all dimensions starting from dim"
(used for pairwise distance computation).
"""

import torch
from torch import Tensor
from typing import Optional


def l0_distances(x: Tensor, x_adv: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Compute L0 distance (number of perturbed elements) between original and adversarial samples.
    
    Args:
        x: Original inputs, shape (batch_size, ...)
        x_adv: Adversarial inputs, shape (batch_size, ...)
        dim: Starting dimension from which to flatten and compute distance.
             If None, flattens from dim 1.
        
    Returns:
        L0 distances per sample
    """
    diff = (x - x_adv).abs()
    if dim is not None:
        # Flatten from dim onwards and reduce over the last dimension
        return (diff.flatten(start_dim=dim) > 1e-10).sum(dim=-1).float()
    else:
        return (diff.flatten(start_dim=1) > 1e-10).sum(dim=1).float()


def l1_distances(x: Tensor, x_adv: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Compute L1 distance (sum of absolute differences) between original and adversarial samples.
    
    Args:
        x: Original inputs, shape (batch_size, ...)
        x_adv: Adversarial inputs, shape (batch_size, ...)
        dim: Starting dimension from which to flatten and compute distance.
             If None, flattens from dim 1.
        
    Returns:
        L1 distances per sample
    """
    diff = (x - x_adv).abs()
    if dim is not None:
        # Flatten from dim onwards and reduce over the last dimension
        return diff.flatten(start_dim=dim).sum(dim=-1)
    else:
        return diff.flatten(start_dim=1).sum(dim=1)


def l2_distances(x: Tensor, x_adv: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Compute L2 (Euclidean) distance between original and adversarial samples.
    
    Args:
        x: Original inputs, shape (batch_size, ...)
        x_adv: Adversarial inputs, shape (batch_size, ...)
        dim: Starting dimension from which to flatten and compute distance.
             If None, flattens from dim 1.
        
    Returns:
        L2 distances per sample
    """
    diff = x - x_adv
    if dim is not None:
        # Flatten from dim onwards and reduce over the last dimension
        return diff.flatten(start_dim=dim).norm(p=2, dim=-1)
    else:
        return diff.flatten(start_dim=1).norm(p=2, dim=1)


def linf_distances(x: Tensor, x_adv: Tensor, dim: Optional[int] = None) -> Tensor:
    """
    Compute L∞ distance (maximum absolute difference) between original and adversarial samples.
    
    Args:
        x: Original inputs, shape (batch_size, ...)
        x_adv: Adversarial inputs, shape (batch_size, ...)
        dim: Starting dimension from which to flatten and compute distance.
             If None, flattens from dim 1.
        
    Returns:
        L∞ distances per sample
    """
    diff = (x - x_adv).abs()
    if dim is not None:
        # Flatten from dim onwards and reduce over the last dimension
        return diff.flatten(start_dim=dim).max(dim=-1)[0]
    else:
        return diff.flatten(start_dim=1).max(dim=1)[0]
