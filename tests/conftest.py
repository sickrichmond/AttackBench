"""Toy model, data and attacks shared by the tests. Everything runs on CPU in milliseconds."""
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FEATURES = (1, 2, 3)  # tiny "images"


class ThresholdNet(nn.Module):
    """
    Predicts class 1 iff the first pixel is > 0.5, with a hard margin.

    An attack only has to push that pixel across 0.5, so the exact minimal distance of
    every sample is known in advance — which is what makes the assertions meaningful.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.flatten(1)[:, 0]
        return torch.stack([0.5 - v, v - 0.5], dim=1) * 100


def make_loader(n: int = 8, batch_size: int = 4) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    inputs = torch.rand((n, *FEATURES), generator=generator)
    # keep the decision pixel away from the boundary so distances are unambiguous
    inputs[:, 0, 0, 0] = torch.linspace(0.1, 0.9, n)
    labels = (inputs.flatten(1)[:, 0] > 0.5).long()
    return DataLoader(TensorDataset(inputs, labels), batch_size=batch_size)


def _crossed(inputs: torch.Tensor, margin: float) -> torch.Tensor:
    """Copy of inputs with the decision pixel moved just across the 0.5 boundary."""
    out = inputs.clone()
    v = out[:, 0, 0, 0]
    out[:, 0, 0, 0] = torch.where(v > 0.5, 0.5 - margin, 0.5 + margin)
    return out


@pytest.fixture
def loader():
    return make_loader()


@pytest.fixture
def model():
    return ThresholdNet()


def tight_attack(model, inputs, labels, margin=0.01, **kwargs):
    """Succeeds on every sample with the smallest perturbation we allow ourselves."""
    return _crossed(inputs, margin)


def loose_attack(model, inputs, labels, **kwargs):
    """Succeeds too, but always with a bigger perturbation than tight_attack."""
    return _crossed(inputs, 0.2)


def lazy_attack(model, inputs, labels, **kwargs):
    """Returns the input untouched: never succeeds."""
    return inputs.clone()


def discarding_attack(model, inputs, labels, **kwargs):
    """
    Finds a good adversarial example, queries the model with it, then returns a much
    worse one — the behaviour AttackBench's d* tracking exists to catch.
    """
    good = _crossed(inputs, 0.01)
    model(good)  # tracked: this is where d* comes from
    return _crossed(inputs, 0.4)


def wasteful_attack(model, inputs, labels, n_queries=200, **kwargs):
    """Burns queries without ever succeeding, to exercise the budget."""
    for _ in range(n_queries):
        model(inputs)
    return inputs.clone()
