"""
AttackBench - A Python package for benchmarking adversarial attacks.

Usage::

    import attackbench

    results = attackbench.run_attack(model, dataset, attack, 'linf', device)
    stats = attackbench.get_stats(results, 'linf')

Attacks run under a query budget of DEFAULT_QUERY_BUDGET forward+backward propagations
per sample, and ``results['distances']`` holds the smallest perturbation found during
the optimization — the two things that make runs comparable with the AttackBench paper.

Optional subpackages, independent of each other:

- ``attacks``: ``pip install attackbenchlib[attacks]`` (ART, Foolbox, CleverHans)
- ``torchattacks``: isolated extra for Torchattacks' legacy dependency stack
- ``models``: ``pip install attackbenchlib[models]`` (RobustBench model zoo)
- ``metrics``: ``pip install attackbenchlib[metrics]`` (analysis & evaluation tools)
"""

import importlib
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("attackbenchlib")
except PackageNotFoundError:  # source checkout without an install
    __version__ = "0+unknown"

# ── Core API (always available) ──────────────────────────────────────────
from .run import run_attack, DEFAULT_QUERY_BUDGET
from .custom_components import create_custom_attack

# ── Helpers to load objects (always available) ───────────────────────────
from .datasets.registry import get_loader


# ── RobustBench integration ─────────────────────────────────────────────
def load_model(model_name, dataset='cifar10', threat_model='Linf', **kwargs):
    """Load a RobustBench model and attach AttackBench metadata for automatic extraction."""
    try:
        from .models.robustbench_compat import get_robustbench_loader

        _rb_load_model = get_robustbench_loader()
    except ModuleNotFoundError as e:
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
    upload_optimal_distances,
    download_optimal_distances,
    update_optimal_distances,
)

# ── Lazy imports for optional subpackages (attacks/, models/, metrics/) ──
# Resolved on first access via __getattr__ (PEP 562); if the subpackage is not
# installed, a clear ImportError naming the pip extra is raised.
_LAZY = {
    # name: (submodule, attribute, pip extra)
    'get_model':                   ('models.registry',           'get_model',                  'models'),
    'get_attack':                  ('attacks.registry',          'get_attack',                 'attacks'),
    'list_attacks':                ('attacks.registry',          'list_attacks',               'attacks'),
    'bomn_attack':                 ('attacks.bomn',              'bomn_attack',                'attacks'),
    'get_stats':                   ('metrics.analysis',          'get_stats',                  'metrics'),
    'compare_attacks':             ('metrics.analysis',          'compare_attacks',            'metrics'),
    'compute_curves':              ('metrics.analysis',          'compute_curves',             'metrics'),
    'compute_optimality':          ('metrics.analysis',          'compute_optimality',         'metrics'),
    'compute_efficiency':          ('metrics.analysis',          'compute_efficiency',         'metrics'),
    'eval_optimality':             ('metrics.distances',         'eval_optimality',            'metrics'),
    'ensemble_gain':               ('metrics.ensemble',          'ensemble_gain',              'metrics'),
    'ensemble_distances':          ('metrics.ensemble',          'ensemble_distances',         'metrics'),
    'compute_local_optimality':    ('metrics.optimality',        'compute_local_optimality',   'metrics'),
    'compare_attacks_optimality':  ('metrics.optimality',        'compare_attacks_optimality', 'metrics'),
    'compute_global_optimality':   ('metrics.global_optimality', 'compute_global_optimality',  'metrics'),
    'create_attack_leaderboard':   ('metrics.global_optimality', 'create_attack_leaderboard',  'metrics'),
    'compare_attacks_global':      ('metrics.global_optimality', 'compare_attacks_global',     'metrics'),
    'format_leaderboard':          ('metrics.global_optimality', 'format_leaderboard',         'metrics'),
}


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'attackbench' has no attribute '{name}'")

    submodule, attr, extra = _LAZY[name]
    try:
        value = getattr(importlib.import_module(f'.{submodule}', __name__), attr)
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            f"attackbench.{name} requires the '{extra}' subpackage. "
            f"Install it with: pip install attackbenchlib[{extra}]"
        ) from e

    globals()[name] = value  # cache for subsequent accesses
    return value


def __dir__():
    """Support tab-completion for lazy attributes."""
    return list(globals().keys()) + list(_LAZY.keys())


__all__ = [
    'run_attack',
    'DEFAULT_QUERY_BUDGET',
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
    'upload_optimal_distances',
    'download_optimal_distances',
    'update_optimal_distances',
]
