"""
AttackBench - A Python package for benchmarking adversarial attacks.

Usage:
    import attackbench

    results = attackbench.run_attack(model, dataset, attack, 'linf', device)
    stats = attackbench.get_stats(results, 'linf')

Optional subpackages:
    - attacks: pip install attackbenchlib[attacks]  (adversarial attack libraries)
    - metrics: pip install attackbenchlib[metrics]  (analysis & evaluation tools)
    Both are optional and independent of each other.
"""

import importlib

# Version info
__version__ = "1.0.0"

# ── Core API (always available) ──────────────────────────────────────────
from .run import run_attack
from .custom_components import create_custom_attack

# ── Helpers to load objects (always available) ───────────────────────────
from .datasets.registry import get_loader


# ── RobustBench integration ─────────────────────────────────────────────
def load_model(model_name, dataset='cifar10', threat_model='Linf', **kwargs):
    """Load a RobustBench model and attach AttackBench metadata for automatic extraction."""
    try:
        from robustbench import load_model as _rb_load_model
    except ModuleNotFoundError as e:
        if 'autoattack' in str(e):
            raise ImportError(
                "robustbench requires 'autoattack', which failed to install from PyPI.\n"
                "Fix with: pip install git+https://github.com/fra31/auto-attack"
            ) from e
        raise ImportError(
            "robustbench is required to load models. "
            "Install it with: pip install attackbenchlib[models]"
        ) from e
    model = _rb_load_model(model_name=model_name, dataset=dataset, threat_model=threat_model, **kwargs)
    model._attackbench_model = model_name
    model._attackbench_dataset = dataset
    return model

# ── W&B integration ─────────────────────────────────────────────────────
from .wandb import (
    upload_precompiled_distances,
    download_precompiled_distances,
    list_available_distances,
    upload_optimal_distances,
    download_optimal_distances,
    get_precompiled_distances,
    get_optimal_distances,
)

# ── Lazy imports for optional subpackages (attacks/ and metrics/) ────────
# These symbols are resolved on first access via __getattr__ (PEP 562).
# If the subpackage is not installed, a clear ImportError is raised.

_LAZY_ATTACKS = {
    'get_attack':   ('attacks.registry', 'get_attack'),
    'list_attacks': ('attacks.registry', 'list_attacks'),
    'bomn_attack':  ('attacks.bomn',       'bomn_attack'),
}

_LAZY_MODELS = {
    'get_model':    ('models.registry',    'get_model'),
}

_LAZY_METRICS = {
    'get_stats':                   ('metrics.analysis',         'get_stats'),
    'eval_optimality':             ('metrics.distances',        'eval_optimality'),
    'ensemble_gain':               ('metrics.ensemble',         'ensemble_gain'),
    'ensemble_distances':          ('metrics.ensemble',         'ensemble_distances'),
    'compare_attacks':             ('metrics.analysis',         'compare_attacks'),
    'compute_curves':              ('metrics.analysis',         'compute_curves'),
    'compute_optimality':          ('metrics.analysis',         'compute_optimality'),
    'compute_efficiency':          ('metrics.analysis',         'compute_efficiency'),
    'compute_local_optimality':    ('metrics.optimality',       'compute_local_optimality'),
    'compare_attacks_optimality':  ('metrics.optimality',       'compare_attacks_optimality'),
    'compute_global_optimality':   ('metrics.global_optimality','compute_global_optimality'),
    'create_attack_leaderboard':   ('metrics.global_optimality','create_attack_leaderboard'),
    'compare_attacks_global':      ('metrics.global_optimality','compare_attacks_global'),
    'format_leaderboard':          ('metrics.global_optimality','format_leaderboard'),
}


def __getattr__(name: str):
    # ── Models (requires robustbench) ───────────────────────────────────
    if name in _LAZY_MODELS:
        submodule, attr = _LAZY_MODELS[name]
        try:
            mod = importlib.import_module(f'.{submodule}', __name__)
            value = getattr(mod, attr)
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"attackbench.{name} requires robustbench. "
                f"Install it with: pip install attackbenchlib[models]"
            ) from e
        globals()[name] = value
        return value

    # ── Attacks (optional) ───────────────────────────────────────────────
    if name in _LAZY_ATTACKS:
        submodule, attr = _LAZY_ATTACKS[name]
        try:
            mod = importlib.import_module(f'.{submodule}', __name__)
            value = getattr(mod, attr)
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"attackbench.{name} requires the 'attacks' subpackage. "
                f"Install it with: pip install attackbenchlib[attacks]"
            ) from e
        globals()[name] = value          # cache for subsequent accesses
        return value

    # ── Metrics (optional) ───────────────────────────────────────────────
    if name in _LAZY_METRICS:
        submodule, attr = _LAZY_METRICS[name]
        try:
            mod = importlib.import_module(f'.{submodule}', __name__)
            value = getattr(mod, attr)
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                f"attackbench.{name} requires the 'metrics' subpackage. "
                f"Install it with: pip install attackbenchlib[metrics]"
            ) from e
        globals()[name] = value
        return value

    raise AttributeError(f"module 'attackbench' has no attribute '{name}'")


def __dir__():
    """Support tab-completion for lazy attributes."""
    public = list(globals().keys())
    public.extend(_LAZY_MODELS.keys())
    public.extend(_LAZY_ATTACKS.keys())
    public.extend(_LAZY_METRICS.keys())
    return public


__all__ = [
    'run_attack',
    'get_stats',

    # Helpers to load objects
    'load_model',
    'get_model',
    'get_loader',
    'get_attack',
    'list_attacks',

    # Custom components
    'create_custom_attack',

    # BoMN composite attack
    'bomn_attack',

    # Analysis functions
    'eval_optimality',
    'ensemble_gain',
    'ensemble_distances',
    'compare_attacks',
    'compute_curves',
    'compute_optimality',
    'compute_efficiency',

    # Stage 3: Local Optimality
    'compute_local_optimality',
    'compare_attacks_optimality',

    # Stage 4-5: Global Optimality & Ranking
    'compute_global_optimality',
    'create_attack_leaderboard',
    'compare_attacks_global',
    'format_leaderboard',

    # W&B functions
    'upload_precompiled_distances',
    'download_precompiled_distances',
    'list_available_distances',
    'get_precompiled_distances',
    'upload_optimal_distances',
    'download_optimal_distances',
    'get_optimal_distances',
]
