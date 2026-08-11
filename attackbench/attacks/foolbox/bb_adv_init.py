from foolbox.attacks import L0BrendelBethgeAttack, L1BrendelBethgeAttack, L2BrendelBethgeAttack, LinfinityBrendelBethgeAttack
from torch import Tensor
from typing import Callable
from foolbox.attacks.dataset_attack import DatasetAttack
from foolbox import PyTorchModel

_bb_attacks = {
    'l0': L0BrendelBethgeAttack,
    'l1': L1BrendelBethgeAttack,
    'l2': L2BrendelBethgeAttack,
    'linf': LinfinityBrendelBethgeAttack,
}

def dataset_BB_attack(bb_attack: Callable, model: PyTorchModel, inputs: Tensor, labels: Tensor, **kwargs):
    dataset_atk = DatasetAttack()
    dataset_atk.feed(model, inputs)
    atk = bb_attack(init_attack=dataset_atk, **kwargs)
    _, advs, success = atk(model, inputs, labels, epsilons=None)
    return advs
