"""
AttackBench - A Python package for benchmarking adversarial attacks.

Usage:
    import attackbench

    results = attackbench.run_attack(model, dataset, attack, 'linf', device)
    stats = attackbench.get_stats(results, 'linf')
"""

# Version info
__version__ = "1.0.0"

# ── Core API ─────────────────────────────────────────────────────────────
from .run import run_attack, get_stats
from .custom_components import create_custom_attack

# ── Helpers to load objects ──────────────────────────────────────────────
from .attacks.ingredient import get_attack
from .datasets.ingredient import get_loader
from .models.ingredient import get_model

# ── BoMN (Best-of-MinNorm) composite attack ─────────────────────────────
from .attacks.bomn import bomn_attack

# ── RobustBench integration ─────────────────────────────────────────────
def load_model(model_name, dataset='cifar10', threat_model='Linf', **kwargs):
    """Load a RobustBench model and attach AttackBench metadata for automatic extraction."""
    from robustbench import load_model as _rb_load_model
    model = _rb_load_model(model_name=model_name, dataset=dataset, threat_model=threat_model, **kwargs)
    model._attackbench_model = model_name
    model._attackbench_dataset = dataset
    return model

# ── Analysis functions (from metrics) ────────────────────────────────────
from .metrics import eval_optimality, ensemble_gain, ensemble_distances

# ── Stage 3: Local Optimality ────────────────────────────────────────────
def compute_local_optimality(*args, **kwargs):
    from .metrics import compute_local_optimality as _compute_local_optimality
    return _compute_local_optimality(*args, **kwargs)

def compare_attacks_optimality(*args, **kwargs):
    from .metrics import compare_attacks_optimality as _compare_attacks_optimality
    return _compare_attacks_optimality(*args, **kwargs)

# ── Stage 4-5: Global Optimality & Ranking ───────────────────────────────
def compute_global_optimality(*args, **kwargs):
    from .metrics import compute_global_optimality as _compute_global_optimality
    return _compute_global_optimality(*args, **kwargs)

def create_attack_leaderboard(*args, **kwargs):
    from .metrics import create_attack_leaderboard as _create_attack_leaderboard
    return _create_attack_leaderboard(*args, **kwargs)

def compare_attacks_global(*args, **kwargs):
    from .metrics import compare_attacks_global as _compare_attacks_global
    return _compare_attacks_global(*args, **kwargs)

def format_leaderboard(*args, **kwargs):
    from .metrics import format_leaderboard as _format_leaderboard
    return _format_leaderboard(*args, **kwargs)

# ── W&B integration ─────────────────────────────────────────────────────
from .wandb import (
    upload_precompiled_distances,
    download_precompiled_distances,
    upload_directory,
    list_available_distances,
    upload_optimal_distances,
    download_optimal_distances,
    get_precompiled_distances,
    get_optimal_distances,
)

__all__ = [
    'run_attack',
    'get_stats',

    # Helpers to load objects
    'load_model',
    'get_model',
    'get_loader',
    'get_attack',

    # Custom components
    'create_custom_attack',

    # BoMN composite attack
    'bomn_attack',

    # Analysis functions
    'eval_optimality',
    'ensemble_gain',
    'ensemble_distances',

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
    'upload_directory',
    'list_available_distances',
    'get_precompiled_distances',
    'upload_optimal_distances',
    'download_optimal_distances',
    'get_optimal_distances',
]
