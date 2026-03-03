minimal_search_steps = 20
minimal_init_eps = {
    'l0': 100,
    'l1': 10,
    'l2': 1,
    'linf': 1 / 255,
}

# Pre-configured attack instances — importable as `from attackbench.attacks import pgd`
from ..preconfigured import (
    pgd, fgsm, apgd, fab, fmn, deepfool, superdeepfool, trust_region
)

__all__ = [
    'minimal_search_steps', 'minimal_init_eps',
    'pgd', 'fgsm', 'apgd', 'fab', 'fmn', 'deepfool', 'superdeepfool', 'trust_region',
]