"""
Original AttackBench attacks - directly importable.

Usage:
    from attackbench.attacks import pgd, fgsm, apgd, fab, fmn
    
    # Use directly
    results = attackbench.run_attack(
        model=model,
        dataset=dataset,
        attack=pgd,  # Direct import
        threat_model='linf'
    )
"""

# Import dalla tua implementazione original
from .attacks.original.configs import (
    get_original_apgd,
    get_original_fab,
    get_original_fmn,
    get_original_tr,
    get_original_deepfool,
    get_original_superdeepfool,
)

# Perturbation budgets for the FixedBudget attacks below, per threat model. The usual
# CIFAR-10 conventions; override with run_attack(..., epsilon=...).
_DEFAULT_EPS = {'linf': 8 / 255, 'l2': 0.5, 'l1': 12.0, 'l0': 20.0}


def _fixed_budget(getter, name, **params):
    """
    Build a FixedBudget attack whose epsilon follows the threat model that run_attack
    injects, instead of freezing 8/255 — an linf budget — for every norm.
    """
    def attack(model, inputs, labels, threat_model='linf', epsilon=None,
               targets=None, targeted=False, **kwargs):
        if epsilon is None:
            if threat_model not in _DEFAULT_EPS:
                raise ValueError(f"No default epsilon for threat model '{threat_model}'; "
                                 f"pass epsilon=... to run_attack().")
            epsilon = _DEFAULT_EPS[threat_model]
        attack_fn = getter(threat_model=threat_model, epsilon=epsilon, **params)
        return attack_fn(model=model, inputs=inputs, labels=labels,
                         targets=targets, targeted=targeted, **kwargs)

    attack.__name__ = name
    attack._attackbench_name = name
    attack._attackbench_lib = 'original'
    return attack


# Create attack instances with default parameters
# Note: threat_model is NOT set here — it is injected at runtime
# by run_attack() based on its threat_model parameter.
pgd = _fixed_budget(get_original_apgd, 'pgd',
                    num_steps=40, num_restarts=1, loss='ce', rho=0.75, use_largereps=False)

fgsm = _fixed_budget(get_original_apgd, 'fgsm',  # APGD con 1 step = FGSM-like
                     num_steps=1, num_restarts=1, loss='ce', rho=0.75, use_largereps=False)

apgd = _fixed_budget(get_original_apgd, 'apgd',
                     num_steps=100, num_restarts=1, loss='ce', rho=0.75, use_largereps=False)

fab = _fixed_budget(get_original_fab, 'fab',
                    num_restarts=1, num_steps=100, alpha_max=0.1, eta=1.05, beta=0.9,
                    targeted_variant=False, n_target_classes=9)

fmn = get_original_fmn(
    num_steps=1000, 
    max_step_size=10, 
    gamma=0.05
)
fmn._attackbench_name = 'fmn'
fmn._attackbench_lib = 'original'

deepfool = get_original_deepfool(
    num_classes=10, 
    overshoot=0.02, 
    num_steps=50
)
deepfool._attackbench_name = 'deepfool'
deepfool._attackbench_lib = 'original'

superdeepfool = get_original_superdeepfool(
    num_classes=10,
    overshoot=0.02,
    num_steps=50,
    alpha=1.5,
    adaptive_overshoot=True
)
superdeepfool._attackbench_name = 'superdeepfool'
superdeepfool._attackbench_lib = 'original'

trust_region = get_original_tr(
    adaptive=False, 
    epsilon=0.001, 
    c=9, 
    num_steps=100
)
trust_region._attackbench_name = 'trust_region'
trust_region._attackbench_lib = 'original'

__all__ = ['pgd', 'fgsm', 'apgd', 'fab', 'fmn', 'deepfool', 'superdeepfool', 'trust_region']