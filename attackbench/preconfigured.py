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

# Create attack instances with default parameters
pgd = get_original_apgd(
    threat_model='linf', 
    num_steps=40, 
    num_restarts=1, 
    epsilon=8/255, 
    loss='ce', 
    rho=0.75, 
    use_largereps=False
)
pgd._attackbench_name = 'pgd'
pgd._attackbench_lib = 'original'

fgsm = get_original_apgd(  # APGD con 1 step = FGSM-like
    threat_model='linf', 
    num_steps=1, 
    num_restarts=1, 
    epsilon=8/255, 
    loss='ce', 
    rho=0.75, 
    use_largereps=False
)
fgsm._attackbench_name = 'fgsm'
fgsm._attackbench_lib = 'original'

apgd = get_original_apgd(
    threat_model='linf', 
    num_steps=100, 
    num_restarts=1, 
    epsilon=8/255, 
    loss='ce', 
    rho=0.75, 
    use_largereps=False
)
apgd._attackbench_name = 'apgd'
apgd._attackbench_lib = 'original'

fab = get_original_fab(
    threat_model='linf', 
    num_restarts=1, 
    num_steps=100, 
    epsilon=8/255, 
    alpha_max=0.1, 
    eta=1.05, 
    beta=0.9, 
    targeted_variant=False, 
    n_target_classes=9
)
fab._attackbench_name = 'fab'
fab._attackbench_lib = 'original'

fmn = get_original_fmn(
    threat_model='linf', 
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
    threat_model='linf', 
    adaptive=False, 
    epsilon=0.001, 
    c=9, 
    num_steps=100
)
trust_region._attackbench_name = 'trust_region'
trust_region._attackbench_lib = 'original'

__all__ = ['pgd', 'fgsm', 'apgd', 'fab', 'fmn', 'deepfool', 'superdeepfool', 'trust_region']