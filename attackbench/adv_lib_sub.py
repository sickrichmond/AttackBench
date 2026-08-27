"""
adv_lib_sub.py - Substitute for adv_lib external dependency.

This module provides internal implementations of functions that were
previously imported from adv_lib, allowing AttackBench to work without
the external adv_lib dependency.

Contains:
- Distance metrics (l0, l1, l2, linf)
- Default metrics dictionary
- Loss functions (difference_of_logits)
- Model utilities (normalize_model, NormalizeLayer)
"""

from collections import OrderedDict
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

# =============================================================================
# DISTANCE METRICS
# =============================================================================


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
        return diff.flatten(start_dim=dim).max(dim=-1)[0]
    else:
        return diff.flatten(start_dim=1).max(dim=1)[0]


# =============================================================================
# DEFAULT METRICS
# =============================================================================

# Default metrics dictionary for tracking distance metrics
# This matches the adv_lib._default_metrics structure
_default_metrics = OrderedDict(
    [
        ("linf", linf_distances),
        ("l2", l2_distances),
        ("l1", l1_distances),
        ("l0", l0_distances),
    ]
)


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


def difference_of_logits(
    logits: Tensor, labels: Tensor, targeted: bool = False
) -> Tensor:
    """
    Compute the Difference of Logits (DL) loss.

    DL loss is defined as:
    - For untargeted: logit[true_class] - max(logit[other_classes])
    - For targeted: max(logit[other_classes]) - logit[target_class]

    Positive values mean the attack objective has not been reached yet. Values at or
    below zero indicate success (misclassification for untargeted attacks, or the
    requested class becoming maximal for targeted attacks).

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

    # Masking by multiplication would evaluate ``0 * -inf`` at every non-label
    # position and turn an otherwise finite loss into NaN. masked_fill avoids that
    # undefined arithmetic and preserves finite gradients.
    label_mask = torch.zeros_like(logits, dtype=torch.bool)
    label_mask.scatter_(1, labels.unsqueeze(1), True)
    other_logits = logits.masked_fill(label_mask, float("-inf")).max(dim=1).values

    if targeted:
        # Minimise until the target logit is at least the largest other logit.
        return other_logits - target_logits
    else:
        # Minimise until another logit is at least the true-class logit.
        return target_logits - other_logits


# =============================================================================
# MODEL UTILITIES
# =============================================================================


class NormalizeLayer(nn.Module):
    """Normalization layer to be prepended to a model."""

    def __init__(
        self,
        mean: Union[Tuple[float, ...], Tensor],
        std: Union[Tuple[float, ...], Tensor],
    ):
        """
        Initialize normalization layer.

        Args:
            mean: Mean values for each channel
            std: Standard deviation values for each channel
        """
        super(NormalizeLayer, self).__init__()

        if isinstance(mean, (tuple, list)):
            mean = torch.tensor(mean)
        if isinstance(std, (tuple, list)):
            std = torch.tensor(std)

        self.register_buffer("mean", mean.view(1, -1, 1, 1))
        self.register_buffer("std", std.view(1, -1, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize input tensor."""
        return (x - self.mean) / self.std


def normalize_model(
    model: nn.Module,
    mean: Union[Tuple[float, ...], Tensor],
    std: Union[Tuple[float, ...], Tensor],
) -> nn.Module:
    """
    Prepend a normalization layer to a model.

    Creates a sequential model with normalization as the first layer,
    allowing the model to accept [0, 1] normalized inputs.

    Args:
        model: PyTorch model to wrap
        mean: Mean values for normalization (per channel)
        std: Standard deviation values for normalization (per channel)

    Returns:
        Sequential model with normalization prepended

    Example:
        >>> model = resnet18()
        >>> # Normalize with ImageNet stats
        >>> normalized_model = normalize_model(
        ...     model,
        ...     mean=(0.485, 0.456, 0.406),
        ...     std=(0.229, 0.224, 0.225)
        ... )
    """
    normalize_layer = NormalizeLayer(mean, std)
    return nn.Sequential(normalize_layer, model)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Distance metrics
    "l0_distances",
    "l1_distances",
    "l2_distances",
    "linf_distances",
    # Default metrics
    "_default_metrics",
    # Loss functions
    "difference_of_logits",
    # Model utilities
    "NormalizeLayer",
    "normalize_model",
]
