from attack_evaluation.run import run_attack, get_stats  # Import from run.py
# OR if you prefer from metrics package:
# from attack_evaluation.metrics.analysis import get_stats

from attack_evaluation.custom_components import create_custom_attack

# Import con nuova signature
from attack_evaluation.attacks.ingredient import get_attack  # DIRETTO
from attack_evaluation.datasets.ingredient import get_loader  
from attack_evaluation.models.ingredient import get_model

# BoMN (Best-of-MinNorm) composite attack
from attack_evaluation.attacks.bomn import bomn_attack

# RobustBench integration
from robustbench import load_model

# EXPORT EXISTING ANALYSIS FUNCTIONS
try:
    from analysis.utils import eval_optimality, ensemble_gain, ensemble_distances
    from analysis.plot_distances import plot_robust_accuracy_curve
    from analysis.compile_jsons import load_precompiled_distances, compare_attacks
    analysis_available = True
except ImportError:
    analysis_available = False

# W&B integration for precompiled distances
from .wandb_manager import (
    upload_precompiled_distances,
    download_precompiled_distances, 
    upload_directory,
    list_available_distances
)

# W&B utils for database reading
from .wandb_utils import get_precompiled_distances

__all__ = [
    'run_attack',
    'get_stats',  # NUOVO
    
    # Helpers to load objects
    'load_model',
    'get_model',
    'get_loader', 
    'get_attack',  # SEMPLIFICATO
    
    # Custom components
    'create_custom_attack',
    
    # BoMN composite attack
    'bomn_attack',
    'create_bomn',
    
    # W&B functions
    'upload_precompiled_distances',
    'download_precompiled_distances',
    'upload_directory', 
    'list_available_distances',
    'get_precompiled_distances'
]

if analysis_available:
    __all__.extend([
        'eval_optimality',
        'ensemble_gain', 
        'ensemble_distances',
        'plot_robust_accuracy_curve',
        'load_precompiled_distances',
        'compare_attacks'
    ])