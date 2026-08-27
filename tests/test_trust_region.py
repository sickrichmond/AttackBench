import pytest
import torch

from attackbench.attacks.original.trust_region import tr_attack
from conftest import ThresholdNet


@pytest.mark.parametrize("threat_model", ["l2", "linf"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_trust_region_crosses_the_boundary_and_restores_model_state(
    threat_model, adaptive
):
    model = ThresholdNet()
    model.train()
    inputs = torch.zeros(2, 1, 2, 3)
    inputs[0, 0, 0, 0] = 0.1
    inputs[1, 0, 0, 0] = 0.9
    labels = model(inputs).argmax(dim=1)

    adversarial = tr_attack(
        model,
        inputs,
        labels,
        threat_model,
        adaptive=adaptive,
        eps=0.15,
        c=9,
        iter=6,
    )

    assert model.training
    assert torch.equal(model(adversarial).argmax(dim=1), 1 - labels)
    assert torch.all((0 <= adversarial) & (adversarial <= 1))
    assert torch.equal(inputs[:, :, 1], torch.zeros_like(inputs[:, :, 1]))


def test_trust_region_supports_targeted_attacks():
    model = ThresholdNet()
    inputs = torch.zeros(2, 1, 2, 3)
    inputs[0, 0, 0, 0] = 0.1
    inputs[1, 0, 0, 0] = 0.9
    labels = model(inputs).argmax(dim=1)
    targets = 1 - labels

    adversarial = tr_attack(
        model,
        inputs,
        labels,
        "linf",
        targets=targets,
        targeted=True,
        eps=0.15,
        iter=6,
    )

    assert torch.equal(model(adversarial).argmax(dim=1), targets)


def test_trust_region_rejects_invalid_configuration():
    model = ThresholdNet()
    inputs = torch.zeros(1, 1, 2, 3)
    labels = torch.zeros(1, dtype=torch.long)

    with pytest.raises(ValueError, match="targets are required"):
        tr_attack(model, inputs, labels, "linf", targeted=True)
    with pytest.raises(ValueError, match="Unsupported threat model"):
        tr_attack(model, inputs, labels, "l1")
    with pytest.raises(ValueError, match="eps must be positive"):
        tr_attack(model, inputs, labels, "l2", eps=0)
