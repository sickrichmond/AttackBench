"""Independent sparse projected-gradient attack for the L0 threat model.

The implementation follows the optimization problem in Croce and Hein, "Sparse
and Imperceivable Adversarial Attacks", ICCV 2019. It is implemented directly in
PyTorch and does not derive from the authors source code.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _project_l0(original: Tensor, proposed: Tensor, budgets: Tensor) -> Tensor:
    """Project each sample onto an element-wise L0 ball around ``original``."""
    flat_delta = (proposed - original).flatten(1)
    feature_count = flat_delta.shape[1]
    budgets = budgets.to(device=original.device, dtype=torch.long).clamp(
        min=0, max=feature_count
    )

    order = flat_delta.abs().argsort(dim=1, descending=True)
    sorted_keep = torch.arange(feature_count, device=original.device).unsqueeze(0)
    sorted_keep = sorted_keep < budgets.unsqueeze(1)
    keep = torch.zeros_like(sorted_keep).scatter(1, order, sorted_keep)
    projected = torch.where(keep, flat_delta, torch.zeros_like(flat_delta))
    return (original + projected.reshape_as(original)).clamp(0.0, 1.0)


def _is_successful(
    logits: Tensor,
    labels: Tensor,
    targets: Optional[Tensor],
    targeted: bool,
) -> Tensor:
    predictions = logits.argmax(dim=1)
    if targeted:
        if targets is None:
            raise ValueError("targets are required for a targeted PGD0 attack")
        return predictions.eq(targets)
    return predictions.ne(labels)


def _sparse_pgd(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    budgets: Tensor,
    *,
    num_steps: int,
    step_size: float,
    n_restarts: int,
    targets: Optional[Tensor],
    targeted: bool,
) -> Tensor:
    """Run fixed-budget sparse PGD and return successful points when found."""
    best = inputs.detach().clone()
    with torch.no_grad():
        found = _is_successful(model(inputs), labels, targets, targeted)

    objective_labels = targets if targeted else labels
    if objective_labels is None:
        raise ValueError("targets are required for a targeted PGD0 attack")

    for restart in range(n_restarts):
        if restart == 0:
            current = inputs.detach().clone()
        else:
            random_point = torch.rand_like(inputs)
            current = _project_l0(inputs, random_point, budgets).detach()

        for _ in range(num_steps):
            variable = current.detach().requires_grad_(True)
            logits = model(variable)
            success = _is_successful(logits, labels, targets, targeted)
            newly_found = success & ~found
            best[newly_found] = variable.detach()[newly_found]
            found |= success

            objective = F.cross_entropy(logits, objective_labels, reduction="sum")
            if targeted:
                objective = -objective
            gradient = torch.autograd.grad(objective, variable)[0]
            proposed = variable.detach() + step_size * gradient.sign()
            current = _project_l0(inputs, proposed, budgets).detach()

        with torch.no_grad():
            success = _is_successful(model(current), labels, targets, targeted)
        newly_found = success & ~found
        best[newly_found] = current[newly_found]
        found |= success

    mask_shape = (len(inputs),) + (1,) * (inputs.ndim - 1)
    return torch.where(found.view(mask_shape), best, inputs).detach()


def PGD0_minimal(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    search_steps: int = 10,
    num_steps: int = 100,
    step_size: float = 120000.0 / 255.0,
    kappa: float = -1,
    epsilon: float = -1,
    init_eps: int = 100,
    n_restarts: int = 1,
    targets: Optional[Tensor] = None,
    targeted: bool = False,
) -> Tensor:
    """Search for the smallest successful element-wise L0 budget per sample.

    ``kappa`` and ``epsilon`` remain in the signature for compatibility with the
    1.x configuration schema; the minimal unconstrained-L0 variant does not use
    either parameter.
    """
    if search_steps <= 0:
        raise ValueError("search_steps must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if n_restarts <= 0:
        raise ValueError("n_restarts must be positive")
    if targeted and targets is None:
        raise ValueError("targets are required for a targeted PGD0 attack")

    del kappa, epsilon
    feature_count = inputs[0].numel()
    low = torch.zeros(len(inputs), device=inputs.device, dtype=torch.long)
    current_budget = torch.full_like(low, max(1, int(init_eps))).clamp(
        max=feature_count
    )
    high = current_budget.clone()
    best = inputs.detach().clone()

    with torch.no_grad():
        found_high = _is_successful(model(inputs), labels, targets, targeted)

    for _ in range(search_steps):
        candidate = _sparse_pgd(
            model,
            inputs,
            labels,
            current_budget,
            num_steps=num_steps,
            step_size=step_size,
            n_restarts=n_restarts,
            targets=targets,
            targeted=targeted,
        )
        with torch.no_grad():
            success = _is_successful(model(candidate), labels, targets, targeted)

        improved = success & (~found_high | (current_budget < high))
        best[improved] = candidate[improved]
        high = torch.where(success, current_budget, high)
        low = torch.where(success, low, current_budget)
        found_high |= success

        midpoint = torch.div(low + high, 2, rounding_mode="floor")
        doubled = (current_budget * 2).clamp(max=feature_count)
        current_budget = torch.where(found_high, midpoint, doubled)

        resolved = found_high & ((high - low) <= 1)
        exhausted = ~found_high & (current_budget == feature_count)
        if bool((resolved | exhausted).all()):
            break

    mask_shape = (len(inputs),) + (1,) * (inputs.ndim - 1)
    return torch.where(found_high.view(mask_shape), best, inputs).detach()
