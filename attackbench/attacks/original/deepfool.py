"""Independent PyTorch implementation of the DeepFool algorithm.

The implementation follows the linearized decision-boundary method described in
Moosavi-Dezfooli et al., "DeepFool: a simple and accurate method to fool deep
neural networks", CVPR 2016. It does not derive from the authors' source code.
"""

from typing import Optional

import torch
from torch import Tensor, nn


def _linearized_step(
    model: nn.Module,
    point: Tensor,
    label: int,
    num_classes: int,
) -> Optional[Tensor]:
    """Return the shortest local L2 step to a competing class hyperplane."""
    variable = point.detach().requires_grad_(True)
    logits = model(variable.unsqueeze(0))[0]
    if logits.argmax().item() != label:
        return None

    candidate_count = min(max(2, num_classes), logits.numel())
    candidates = logits.topk(candidate_count).indices.tolist()
    if label not in candidates:
        candidates[-1] = label

    label_gradient = torch.autograd.grad(logits[label], variable, retain_graph=True)[0]
    numerical_eps = torch.finfo(variable.dtype).eps
    shortest_distance = torch.full((), torch.inf, device=point.device)
    shortest_step = None

    for candidate in candidates:
        if candidate == label:
            continue
        candidate_gradient = torch.autograd.grad(
            logits[candidate], variable, retain_graph=True
        )[0]
        normal = candidate_gradient - label_gradient
        squared_norm = normal.flatten().square().sum().clamp_min(numerical_eps)
        score_gap = (logits[candidate] - logits[label]).detach().abs()
        distance = score_gap / squared_norm.sqrt()
        if distance < shortest_distance:
            shortest_distance = distance
            shortest_step = (score_gap + numerical_eps) * normal / squared_norm

    return shortest_step


def deepfool_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    num_classes: int = 10,
    overshoot: float = 0.02,
    max_iter: int = 50,
    targets: Optional[Tensor] = None,
    targeted: bool = False,
    **kwargs,
) -> Tensor:
    """Find untargeted, approximately minimum-L2 adversarial examples.

    Samples that are already misclassified are returned unchanged. DeepFool's
    multiclass construction is untargeted; callers requesting a targeted attack
    receive an explicit error instead of silently running a different protocol.
    """
    if targeted:
        raise ValueError("DeepFool is an untargeted attack")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if max_iter < 0:
        raise ValueError("max_iter must be non-negative")

    del targets, kwargs
    adversarials = inputs.detach().clone()

    for index in range(len(inputs)):
        original = inputs[index].detach()
        label = int(labels[index])
        total_step = torch.zeros_like(original)
        candidate = original

        with torch.no_grad():
            if model(original.unsqueeze(0)).argmax(dim=1).item() != label:
                continue

        for _ in range(max_iter):
            boundary_step = _linearized_step(
                model, candidate, label=label, num_classes=num_classes
            )
            if boundary_step is None:
                break

            total_step = total_step + boundary_step
            candidate = (original + (1.0 + overshoot) * total_step).clamp(0.0, 1.0)
            with torch.no_grad():
                if model(candidate.unsqueeze(0)).argmax(dim=1).item() != label:
                    break

        adversarials[index] = candidate.detach()

    return adversarials
