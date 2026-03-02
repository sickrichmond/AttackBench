# Lazy imports to avoid circular dependencies
def run_attack(*args, **kwargs):
    """Run an adversarial attack. See attack_evaluation.run.run_attack for details."""
    from attack_evaluation.run import run_attack as _run_attack
    return _run_attack(*args, **kwargs)

def get_stats(*args, **kwargs):
    """Get stats from attack results. See attack_evaluation.run.get_stats for details."""
    from attack_evaluation.run import get_stats as _get_stats
    return _get_stats(*args, **kwargs)

# Version info
__version__ = "1.0.0"

from attack_evaluation.custom_components import create_custom_attack

# Import con nuova signature
from attack_evaluation.attacks.ingredient import get_attack  # DIRETTO
from attack_evaluation.datasets.ingredient import get_loader  
from attack_evaluation.models.ingredient import get_model

# BoMN (Best-of-MinNorm) composite attack
from attack_evaluation.attacks.bomn import bomn_attack

# RobustBench integration — wrapped to attach metadata for auto-extraction in run_attack
def load_model(model_name, dataset='cifar10', threat_model='Linf', **kwargs):
    """Load a RobustBench model and attach AttackBench metadata for automatic extraction."""
    from robustbench import load_model as _rb_load_model
    model = _rb_load_model(model_name=model_name, dataset=dataset, threat_model=threat_model, **kwargs)
    # Attach metadata so _extract_metadata() in run_attack can read them automatically
    model._attackbench_model = model_name
    model._attackbench_dataset = dataset
    return model

# Analysis functions — exposed as clean API from attack_evaluation.metrics
from attack_evaluation.metrics import eval_optimality, ensemble_gain, ensemble_distances

# W&B integration for precompiled distances
from .wandb_manager import (
    upload_precompiled_distances,
    download_precompiled_distances, 
    upload_directory,
    list_available_distances,
    upload_optimal_distances,
    download_optimal_distances,
)

# Stage 3: Optimality computation (lazy import to avoid circular dependency)
def compute_local_optimality(*args, **kwargs):
    from attack_evaluation.metrics import compute_local_optimality as _compute_local_optimality
    return _compute_local_optimality(*args, **kwargs)

def compare_attacks_optimality(*args, **kwargs):
    from attack_evaluation.metrics import compare_attacks_optimality as _compare_attacks_optimality
    return _compare_attacks_optimality(*args, **kwargs)

# Stage 4-5: Global Optimality & Ranking (lazy import to avoid circular dependency)
def compute_global_optimality(*args, **kwargs):
    from attack_evaluation.metrics import compute_global_optimality as _compute_global_optimality
    return _compute_global_optimality(*args, **kwargs)

def create_attack_leaderboard(*args, **kwargs):
    from attack_evaluation.metrics import create_attack_leaderboard as _create_attack_leaderboard
    return _create_attack_leaderboard(*args, **kwargs)

def compare_attacks_global(*args, **kwargs):
    from attack_evaluation.metrics import compare_attacks_global as _compare_attacks_global
    return _compare_attacks_global(*args, **kwargs)

def format_leaderboard(*args, **kwargs):
    from attack_evaluation.metrics import format_leaderboard as _format_leaderboard
    return _format_leaderboard(*args, **kwargs)

# W&B utils for database reading
from .wandb_utils import get_precompiled_distances, get_optimal_distances

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

    # Analysis functions (from attack_evaluation.metrics)
    'eval_optimality',
    'ensemble_gain',
    'ensemble_distances',

    # Stage 3: Optimality (API-level)
    'compute_local_optimality',
    'compare_attacks_optimality',

    # Stage 4-5: Global Optimality & Ranking (API-level)
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
    # Optimal distances (lower envelope)
    'upload_optimal_distances',
    'download_optimal_distances',
    'get_optimal_distances',
]