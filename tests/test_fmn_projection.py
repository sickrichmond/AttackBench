"""Regression tests for the independently implemented L1 projection."""

import pytest
import torch

pytest.importorskip("eagerpy")
pytest.importorskip("foolbox")
import eagerpy as ep

from attackbench.attacks.original.fast_minimum_norm import project_onto_l1_ball


def test_l1_projection_respects_radii_and_preserves_feasible_points():
    values = torch.tensor([[3.0, -1.0, 0.5], [0.1, -0.2, 0.3]])
    radii = torch.tensor([2.0, 1.0])

    projected = project_onto_l1_ball(ep.astensor(values), ep.astensor(radii)).raw

    assert torch.allclose(projected[0].abs().sum(), radii[0], atol=1e-6)
    assert torch.equal(projected[1], values[1])
