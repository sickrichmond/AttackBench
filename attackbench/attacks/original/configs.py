from functools import partial
from typing import Callable, Optional

from .auto_pgd import apgd_attack, apgd_minimal_wrapper, apgd_t_attack
from .deepfool import deepfool_attack
from .superdeepfool import superdeepfool_attack
from .fast_adaptive_boundary import fab_attack
from .fast_minimum_norm import fmn_attack
from .pgd_lzero import PGD0_minimal
from .sigma_zero import sigma_zero
from .trust_region import tr_attack
from .. import minimal_init_eps, minimal_search_steps

_prefix = 'original'


def _wrapper(attack, **kwargs): 
    return attack(**kwargs)


def original_apgd():
    return dict(
        name='apgd',
        source='original',
        threat_model='linf',  # available: 'l1', 'l2', 'linf'
        num_steps=100,
        num_restarts=1,
        epsilon=0.3,
        loss='ce',  # loss function in ['ce', 'dlr']
        rho=.75,
        use_largereps=False,  # set True with L1 norm
    )


def original_apgd_l1():
    return dict(
        name='apgd',
        source='original',
        threat_model='l1',
        num_steps=100,
        num_restarts=1,
        epsilon=10,
        loss='ce',  # loss function in ['ce', 'dlr']
        rho=.75,
        use_largereps=True,  # set True with L1 norm
    )


def get_original_apgd(num_steps: int, num_restarts: int, epsilon: float, loss: str, rho: float,
                      use_largereps: bool, threat_model: str = None) -> Callable:
    kwargs = dict(n_iter=num_steps, n_restarts=num_restarts, eps=epsilon,
                  loss=loss, rho=rho, use_largereps=use_largereps)
    if threat_model is not None:
        kwargs['threat_model'] = threat_model
    return partial(apgd_attack, **kwargs)


def original_apgd_minimal():
    return dict(
        name='apgd_minimal',
        source='original',
        threat_model='linf',  # available: 'l1', 'l2', 'linf'
        num_steps=100,
        num_restarts=1,
        loss='ce',  # loss function in ['ce', 'dlr']
        rho=.75,
        use_largereps=False,  # set True with L1 norm
    )


def original_apgd_minimal_l1():
    return dict(
        name='apgd_minimal',
        source='original',
        threat_model='l1',
        num_steps=100,
        num_restarts=1,
        loss='ce',  # loss function in ['ce', 'dlr']
        rho=.75,
        use_largereps=True,  # set True with L1 norm
    )


def get_original_apgd_minimal(threat_model: str, num_steps: int, num_restarts: int, loss: str, rho: float,
                              use_largereps: bool,
                              init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    attack = partial(apgd_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts, loss=loss,
                     rho=rho, use_largereps=use_largereps)
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = 1 if threat_model == 'linf' else None
    return partial(apgd_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def original_apgd_t():
    return dict(
        name='apgd_t',
        source='original',
        threat_model='linf',  # available: 'l1', 'l2', 'linf'
        num_steps=100,
        num_restarts=1,
        num_target_classes=9,
        epsilon=0.3,
        rho=.75,
        use_largereps=False,  # set True with L1 norm
    )


def original_apgd_t_l1():
    return dict(
        name='apgd_t',
        source='original',
        threat_model='l1',
        num_steps=100,
        num_restarts=1,
        num_target_classes=9,
        epsilon=10,
        rho=.75,
        use_largereps=True,  # set True with L1 norm
    )


def get_original_apgd_t(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int, epsilon: float,
                        rho: float, use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                   n_target_classes=num_target_classes, eps=epsilon, rho=rho, use_largereps=use_largereps)


def original_apgd_t_minimal():
    return dict(
        name='apgd_t_minimal',
        source='original',
        threat_model='linf',  # available: 'l1', 'l2', 'linf'
        num_steps=100,
        num_restarts=1,
        num_target_classes=9,
        rho=.75,
        use_largereps=False,  # set True with L1 norm
    )


def original_apgd_t_minimal_l1():
    return dict(
        name='apgd_t_minimal',
        source='original',
        threat_model='l1',
        num_steps=100,
        num_restarts=1,
        num_target_classes=9,
        rho=.75,
        use_largereps=True,  # set True with L1 norm
    )


def get_original_apgd_t_minimal(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int,
                                rho: float, use_largereps: bool,
                                init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    attack = partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                     n_target_classes=num_target_classes, rho=rho, use_largereps=use_largereps)
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = 1 if threat_model == 'linf' else None
    return partial(apgd_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def original_deepfool():
    return dict(
        name='deepfool',
        source='original',
        threat_model='l2',
        num_classes=10,  # number of classes to test gradient (can be different from the number of classes of the model)
        overshoot=0.02,
        num_steps=50,
    )


def get_original_deepfool(num_classes: int, overshoot: float, num_steps: int) -> Callable:
    return partial(deepfool_attack, num_classes=num_classes, overshoot=overshoot, max_iter=num_steps)


def original_fab():
    return dict(
        name='fab',
        source='original',
        threat_model='linf',  # available: 'l1', 'l2', 'linf'
        num_restarts=1,
        num_steps=100,
        epsilon=None,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        targeted_variant=False,
        n_target_classes=9,
    )


def get_original_fab(num_restarts: int, num_steps: int, epsilon: Optional[float], alpha_max: float,
                     eta: float, beta: float, targeted_variant: bool, n_target_classes: int,
                     threat_model: str = None) -> Callable:
    kwargs = dict(n_restarts=num_restarts, n_iter=num_steps, eps=epsilon,
                  alpha_max=alpha_max, eta=eta, beta=beta, targeted_variant=targeted_variant,
                  n_target_classes=n_target_classes)
    if threat_model is not None:
        kwargs['threat_model'] = threat_model
    return partial(fab_attack, **kwargs)


def original_fmn():
    return dict(
        name='fmn',
        source='original',
        threat_model='l2',  # available: 'l0', 'l1', 'l2', 'linf'
        num_steps=1000,
        max_step_size=1,
        gamma=0.05,
    )


def original_fmn_linf():
    return dict(
        name='fmn',
        source='original',
        threat_model='linf',
        num_steps=1000,
        max_step_size=10,
        gamma=0.05,
    )


def get_original_fmn(num_steps: int, max_step_size: float, gamma: float,
                     threat_model: str = None) -> Callable:
    kwargs = dict(steps=num_steps, max_stepsize=max_step_size, gamma=gamma)
    if threat_model is not None:
        kwargs['threat_model'] = threat_model
    return partial(fmn_attack, **kwargs)


def original_tr():
    return dict(
        name='tr',
        source='original',
        threat_model='linf',  # available: 'l2', 'linf'
        adaptive=False,
        epsilon=0.001,
        c=9,
        num_steps=100,
    )


def original_tr_adaptive():
    return dict(
        name='tr',
        source='original',
        threat_model='linf',  # available: 'l2', 'linf'
        adaptive=True,
        epsilon=0.001,
        c=9,
        num_steps=100,
    )


def get_original_tr(adaptive: bool, epsilon: float, c: int, num_steps: int,
                    threat_model: str = None) -> Callable:
    kwargs = dict(adaptive=adaptive, eps=epsilon, c=c, iter=num_steps)
    if threat_model is not None:
        kwargs['threat_model'] = threat_model
    return partial(tr_attack, **kwargs)


def original_sigma_zero():
    return dict(
        name='sigma_zero',
        source='original',
        threat_model='l0',  # available: 'l0', 'l1', 'l2', 'linf'
        num_steps=100,
        lr=1.0,
        sigma=1e-3,
        thr_0=0.3,
        thr_lr=0.01,
        binary_search_steps=10,
    )


def get_original_sigma_zero(threat_model: str, num_steps: int, lr: float, sigma: float, thr_0: float, thr_lr: float,
                            binary_search_steps: int) -> Callable:
    return partial(sigma_zero, steps=num_steps, lr=lr, sigma=sigma, thr_0=thr_0, thr_lr=thr_lr,
                   binary_search_steps=binary_search_steps)


def original_pgd0_minimal():
    return dict(
        name='pgd0_minimal',
        source='original',
        threat_model='l0',
        n_restarts=1,
        num_steps=100,
        step_size=120000 / 255,
        kappa=-1,
        epsilon=-1,
    )


def get_original_pgd0_minimal(threat_model: str, num_steps, step_size, kappa, epsilon, n_restarts,
                              init_eps: Optional[int] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    return partial(PGD0_minimal, search_steps=search_steps, num_steps=num_steps, step_size=step_size, kappa=kappa,
                   epsilon=epsilon, init_eps=init_eps, n_restarts=n_restarts)


def original_superdeepfool():
    return dict(
        name='superdeepfool',
        source='original',
        num_classes=10,
        overshoot=0.02,
        num_steps=50,
        alpha=1.5,
        adaptive_overshoot=True,
    )


def get_original_superdeepfool(num_classes: int, overshoot: float, num_steps: int, 
                                alpha: float = 1.5, adaptive_overshoot: bool = True) -> Callable:
    """
    Get SuperDeepFool attack with specified parameters.
    
    Args:
        num_classes: Number of classes to consider (default: 10)
        overshoot: Initial overshoot parameter (default: 0.02)
        num_steps: Maximum iterations (default: 50)
        alpha: Step size multiplier for faster convergence (default: 1.5)
        adaptive_overshoot: Whether to adaptively adjust overshoot (default: True)
        
    Returns:
        Partial function configured with the specified parameters
    """
    return partial(superdeepfool_attack, num_classes=num_classes, overshoot=overshoot, 
                   num_steps=num_steps, alpha=alpha, adaptive_overshoot=adaptive_overshoot)

