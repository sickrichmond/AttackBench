"""Focused regression tests for release-blocking numerical failures."""

import numpy as np
import torch

from attackbench.adv_lib_sub import difference_of_logits
from attackbench.attacks.original.deepfool import deepfool_attack
from attackbench.attacks.original.superdeepfool import superdeepfool_attack
from attackbench.metrics.curves import compute_robust_accuracy_curve


def test_difference_of_logits_is_finite_and_has_finite_gradients():
    logits = torch.tensor([[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]], requires_grad=True)
    labels = torch.tensor([0, 0])

    untargeted = difference_of_logits(logits, labels)
    targeted = difference_of_logits(logits, labels, targeted=True)

    assert torch.equal(untargeted, torch.tensor([1.0, -2.0]))
    assert torch.equal(targeted, -untargeted)
    assert torch.isfinite(untargeted).all()
    untargeted.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_all_failed_robustness_curve_has_a_finite_domain():
    curve = compute_robust_accuracy_curve(
        distances=np.full(4, np.inf),
        success_mask=np.zeros(4, dtype=bool),
    )

    thresholds = np.asarray(curve["thresholds"])
    assert np.isfinite(thresholds).all()
    assert thresholds[0] == 0
    assert thresholds[-1] == 1
    assert curve["robust_accuracies"] == [1.0] * 100


def test_nonfinite_success_distance_is_not_used_as_a_threshold():
    curve = compute_robust_accuracy_curve(
        distances=np.array([np.inf]), success_mask=np.array([True])
    )
    assert np.isfinite(curve["thresholds"]).all()


def test_native_deepfool_wrappers_preserve_batch_shape_and_box(model):
    inputs = torch.rand(2, 3, 2, 3)
    inputs[:, 0, 0, 0] = torch.tensor([0.25, 0.75])
    labels = (inputs.flatten(1)[:, 0] > 0.5).long()

    deepfool_outputs = deepfool_attack(
        model,
        inputs,
        labels,
        num_classes=2,
        max_iter=3,
    )
    superdeepfool_outputs = superdeepfool_attack(
        model,
        inputs,
        labels,
        num_classes=2,
        num_steps=3,
    )

    for outputs in (deepfool_outputs, superdeepfool_outputs):
        assert outputs.shape == inputs.shape
        assert ((0 <= outputs) & (outputs <= 1)).all()
