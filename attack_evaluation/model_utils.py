"""
Model utility functions.

This module provides utility functions for model manipulation,
replacing dependencies from adv_lib.utils.
"""

import torch
from torch import nn, Tensor
from typing import Tuple, Union


class NormalizeLayer(nn.Module):
    """Normalization layer to be prepended to a model."""
    
    def __init__(self, mean: Union[Tuple[float, ...], Tensor], 
                 std: Union[Tuple[float, ...], Tensor]):
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
            
        self.register_buffer('mean', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))
    
    def forward(self, x: Tensor) -> Tensor:
        """Normalize input tensor."""
        return (x - self.mean) / self.std


def normalize_model(model: nn.Module, 
                    mean: Union[Tuple[float, ...], Tensor],
                    std: Union[Tuple[float, ...], Tensor]) -> nn.Module:
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
