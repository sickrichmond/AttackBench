"""Regression tests for explicit Foolbox device propagation."""

from functools import partial

import pytest
import torch

pytest.importorskip("foolbox")

from attackbench.attacks.foolbox import wrapper
from attackbench.attacks.original import fast_minimum_norm, sigma_zero


class _RecordingModel:
    devices = []

    def __init__(self, model, bounds, device):
        self.devices.append(device)


class _PassthroughAttack:
    def __init__(self, **kwargs):
        pass

    def run(self, model, inputs, criterion, **kwargs):
        return inputs


def test_foolbox_wrapper_uses_input_device(monkeypatch):
    _RecordingModel.devices.clear()
    monkeypatch.setattr(wrapper, "PyTorchModel", _RecordingModel)

    inputs = torch.zeros(2, 1, 2, 2)
    wrapper.foolbox_wrapper(
        partial(_PassthroughAttack), torch.nn.Identity(), inputs, torch.zeros(2, dtype=torch.long)
    )

    assert _RecordingModel.devices == [inputs.device]


def test_fmn_uses_input_device(monkeypatch):
    _RecordingModel.devices.clear()
    monkeypatch.setattr(fast_minimum_norm, "PyTorchModel", _RecordingModel)
    monkeypatch.setitem(fast_minimum_norm._fmn_attacks, "l2", _PassthroughAttack)

    inputs = torch.zeros(2, 1, 2, 2)
    fast_minimum_norm.fmn_attack(
        torch.nn.Identity(), inputs, torch.zeros(2, dtype=torch.long), threat_model="l2"
    )

    assert _RecordingModel.devices == [inputs.device]


def test_sigma_zero_starting_point_uses_input_device(monkeypatch):
    _RecordingModel.devices.clear()
    monkeypatch.setattr(sigma_zero, "PyTorchModel", _RecordingModel)

    class FakeDatasetAttack:
        def feed(self, model, inputs):
            pass

        def __call__(self, model, inputs, labels, epsilons):
            starting_points = torch.ones_like(inputs)
            success = torch.ones(inputs.shape[0], dtype=torch.bool)
            return None, starting_points, success

    class ThresholdModel(torch.nn.Module):
        def forward(self, inputs):
            scores = inputs.flatten(1).mean(1)
            return torch.stack((1 - scores, scores), dim=1)

    monkeypatch.setattr(sigma_zero, "DatasetAttack", FakeDatasetAttack)
    inputs = torch.zeros(2, 1, 2, 2)
    sigma_zero.delta_init(
        ThresholdModel(),
        inputs,
        torch.zeros(2, dtype=torch.long),
        inputs.device,
        starting_point="adversarial",
        binary_search_steps=1,
    )

    assert _RecordingModel.devices == [inputs.device]
