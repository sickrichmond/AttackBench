"""First-order trust-region attack implemented for AttackBench.

The attack repeatedly builds a linear model of a class-margin objective and
restricts each proposed update to a local trust radius. The adaptive variant
uses agreement between predicted and observed margin improvement to resize
that radius.
"""

from typing import Optional

import torch
from torch import Tensor, nn

_SUPPORTED_NORMS = {"l2", "linf"}
_NUMERICAL_EPS = 1e-12


def _dual_norm(gradients: Tensor, threat_model: str) -> Tensor:
    flat = gradients.flatten(1)
    order = 1 if threat_model == "linf" else 2
    return flat.norm(p=order, dim=1)


def _steepest_direction(gradients: Tensor, threat_model: str) -> Tensor:
    if threat_model == "linf":
        return gradients.sign()

    lengths = gradients.flatten(1).norm(p=2, dim=1)
    shape = (len(gradients),) + (1,) * (gradients.ndim - 1)
    return gradients / lengths.clamp_min(_NUMERICAL_EPS).view(shape)


def _perturbation_size(perturbation: Tensor, threat_model: str) -> Tensor:
    flat = perturbation.flatten(1)
    if threat_model == "linf":
        return flat.abs().amax(dim=1)
    return flat.norm(p=2, dim=1)


def _validate_inputs(
    inputs: Tensor,
    labels: Tensor,
    targets: Optional[Tensor],
    targeted: bool,
    threat_model: str,
    eps: float,
    num_candidates: int,
    num_steps: int,
) -> None:
    if threat_model not in _SUPPORTED_NORMS:
        available = ", ".join(sorted(_SUPPORTED_NORMS))
        raise ValueError(
            f"Unsupported threat model {threat_model!r}; choose one of: {available}"
        )
    if inputs.ndim < 2:
        raise ValueError("inputs must have a batch dimension and at least one feature")
    if labels.numel() != len(inputs):
        raise ValueError("labels must contain one class index per input")
    if targeted and targets is None:
        raise ValueError("targets are required when targeted=True")
    if targets is not None and targets.numel() != len(inputs):
        raise ValueError("targets must contain one class index per input")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if num_candidates <= 0:
        raise ValueError("c must be positive")
    if num_steps < 0:
        raise ValueError("iter must be non-negative")


def _select_competitors(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    num_candidates: int,
    threat_model: str,
) -> Tensor:
    """Choose the closest locally linearized non-label class for each input."""
    points = inputs.detach().requires_grad_(True)
    logits = model(points)
    if logits.ndim != 2 or logits.shape[0] != len(inputs):
        raise ValueError("model must return a [batch, classes] logits tensor")
    if logits.shape[1] < 2:
        raise ValueError("trust-region attacks require at least two model classes")
    if labels.min().item() < 0 or labels.max().item() >= logits.shape[1]:
        raise ValueError("labels contain a class index outside the model output")

    candidate_count = min(num_candidates, logits.shape[1] - 1)
    ranked_logits = logits.detach().clone()
    ranked_logits.scatter_(1, labels[:, None], -torch.inf)
    candidates = ranked_logits.topk(candidate_count, dim=1).indices

    rows = torch.arange(len(inputs), device=inputs.device)
    true_logits = logits[rows, labels]
    best_distance = torch.full(
        (len(inputs),), torch.inf, device=inputs.device, dtype=inputs.dtype
    )
    selected = candidates[:, 0].clone()

    for rank in range(candidate_count):
        candidate = candidates[:, rank]
        margin = logits[rows, candidate] - true_logits
        gradients = torch.autograd.grad(
            margin.sum(),
            points,
            retain_graph=rank + 1 < candidate_count,
        )[0]
        estimate = (-margin.detach()).clamp_min(0)
        estimate = estimate / _dual_norm(gradients, threat_model).clamp_min(
            _NUMERICAL_EPS
        )
        better = estimate < best_distance
        best_distance = torch.where(better, estimate, best_distance)
        selected = torch.where(better, candidate, selected)

    return selected.detach()


def _targeted_reference_classes(logits: Tensor, targets: Tensor) -> Tensor:
    """Return the strongest class other than the requested target."""
    competing_logits = logits.detach().clone()
    competing_logits.scatter_(1, targets[:, None], -torch.inf)
    return competing_logits.argmax(dim=1)


def _margin(
    logits: Tensor,
    desired_classes: Tensor,
    reference_classes: Tensor,
) -> Tensor:
    rows = torch.arange(len(logits), device=logits.device)
    return logits[rows, desired_classes] - logits[rows, reference_classes]


def tr_attack(
    model: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    threat_model: str,
    targets: Optional[Tensor] = None,
    targeted: bool = False,
    adaptive: bool = False,
    eps: float = 0.001,
    c: int = 9,
    iter: int = 100,
) -> Tensor:
    """Generate adversarial inputs using first-order trust-region steps.

    eps is the initial per-step trust radius, not a global perturbation budget.
    For untargeted attacks, c controls how many high-logit classes are
    considered when selecting the initial competing class.
    """
    _validate_inputs(inputs, labels, targets, targeted, threat_model, eps, c, iter)

    labels = labels.detach().to(device=inputs.device, dtype=torch.long).reshape(-1)
    if targets is not None:
        targets = (
            targets.detach().to(device=inputs.device, dtype=torch.long).reshape(-1)
        )

    original = inputs.detach()
    current = original.clone()
    best = original.clone()
    best_size = torch.full(
        (len(inputs),), torch.inf, device=inputs.device, dtype=inputs.dtype
    )
    found = torch.zeros(len(inputs), device=inputs.device, dtype=torch.bool)
    radii = torch.full_like(best_size, float(eps))

    max_radius = (
        1.0 if threat_model == "linf" else float(inputs[0].numel()) ** 0.5
    )
    min_radius = max(float(eps) * 1e-3, 1e-7)

    was_training = model.training
    model.eval()
    try:
        if targeted:
            desired_classes = targets
        else:
            desired_classes = _select_competitors(
                model, current, labels, c, threat_model
            )

        radius_shape = (len(inputs),) + (1,) * (inputs.ndim - 1)

        for _ in range(iter):
            points = current.detach().requires_grad_(True)
            logits = model(points)
            predictions = logits.argmax(dim=1)

            if targeted:
                active = predictions.ne(targets)
                reference_classes = _targeted_reference_classes(logits, targets)
            else:
                active = predictions.eq(labels)
                reference_classes = labels

            if not active.any():
                break

            objective = _margin(logits, desired_classes, reference_classes)
            gradients = torch.autograd.grad(objective[active].sum(), points)[0]
            directions = _steepest_direction(gradients, threat_model)

            linear_distance = (-objective.detach()).clamp_min(0)
            linear_distance = linear_distance / _dual_norm(
                gradients, threat_model
            ).clamp_min(_NUMERICAL_EPS)
            proposed_size = torch.minimum(
                radii, linear_distance.mul(1.02).add(1e-6)
            )
            proposed_size = torch.where(
                active, proposed_size, torch.zeros_like(proposed_size)
            )

            candidate = current + directions * proposed_size.view(radius_shape)
            candidate = candidate.clamp(0.0, 1.0).detach()

            with torch.no_grad():
                candidate_logits = model(candidate)
                candidate_predictions = candidate_logits.argmax(dim=1)
                success = (
                    candidate_predictions.eq(targets)
                    if targeted
                    else candidate_predictions.ne(labels)
                )

                sizes = _perturbation_size(candidate - original, threat_model)
                improved = success & (sizes < best_size)
                best[improved] = candidate[improved]
                best_size = torch.where(improved, sizes, best_size)
                found |= success

                if adaptive:
                    actual_gain = _margin(
                        candidate_logits, desired_classes, reference_classes
                    ) - objective.detach()
                    actual_step = candidate - current
                    predicted_gain = (gradients * actual_step).flatten(1).sum(dim=1)
                    valid_prediction = predicted_gain > _NUMERICAL_EPS
                    agreement = torch.full_like(actual_gain, -torch.inf)
                    agreement[valid_prediction] = (
                        actual_gain[valid_prediction]
                        / predicted_gain[valid_prediction]
                    )

                    accepted = active & (actual_gain > 0) & (agreement >= 0.1)
                    current[accepted] = candidate[accepted]

                    shrink = active & (agreement < 0.25)
                    expand = (
                        active
                        & (agreement > 0.75)
                        & (proposed_size >= radii * 0.99)
                    )
                    radii[shrink] *= 0.5
                    radii[expand] *= 2.0
                    radii.clamp_(min=min_radius, max=max_radius)
                else:
                    current[active] = candidate[active]

        return torch.where(found.view(radius_shape), best, current).detach()
    finally:
        model.train(was_training)
