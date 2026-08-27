"""Regression tests for the independent sparse PGD implementation."""

import torch
from torch import nn

from attackbench.attacks.original.pgd_lzero import PGD0_minimal, _project_l0


class _Threshold(nn.Module):
    def forward(self, inputs):
        value = inputs.flatten(1)[:, 0]
        return torch.stack((1.0 - value, value), dim=1)


def test_l0_projection_obeys_per_sample_element_budgets():
    original = torch.zeros(2, 1, 2, 2)
    proposed = torch.tensor([[[[0.9, 0.8], [0.7, 0.6]]], [[[0.9, 0.8], [0.7, 0.6]]]])
    projected = _project_l0(original, proposed, torch.tensor([1, 3]))
    changed = (projected - original).flatten(1).abs().gt(1e-10).sum(dim=1)
    assert changed.tolist() == [1, 3]
    assert ((0 <= projected) & (projected <= 1)).all()


def test_minimal_pgd0_finds_one_feature_adversarials():
    model = _Threshold()
    inputs = torch.tensor([[[[0.25, 0.2]]], [[[0.75, 0.8]]]])
    labels = torch.tensor([0, 1])

    adversarials = PGD0_minimal(
        model,
        inputs,
        labels,
        search_steps=3,
        num_steps=2,
        step_size=1.0,
        init_eps=1,
    )

    assert model(adversarials).argmax(dim=1).ne(labels).all()
    changed = (adversarials - inputs).flatten(1).abs().gt(1e-10).sum(dim=1)
    assert (changed <= 1).all()
