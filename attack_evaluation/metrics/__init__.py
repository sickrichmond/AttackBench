"""
AttackBench Metrics Package

Comprehensive analysis tools for adversarial attack results.
All functions work with raw output from attackbench.run_attack().

Main Functions:
- get_stats(): Complete analysis with granular control
- compute_curves(): Robust accuracy curves only  
- compute_optimality(): Optimality scores only
- compute_efficiency(): Query efficiency only
- compare_attacks(): Multi-attack ensemble analysis
"""

# Main analysis functions
from .analysis import (
    get_stats,
    compare_attacks,
    compute_curves,
    compute_optimality, 
    compute_efficiency
)

# Direct access to component functions
from .distances import (
    compute_distance_statistics,
    eval_optimality,
    compute_basic_metrics,
    compute_attack_efficiency
)

from .curves import (
    compute_robust_accuracy_curve,
    compute_auc_robust_accuracy, 
    compute_certified_robustness_metrics
)

from .ensemble import (
    ensemble_distances,
    complementarity,
    ensemble_gain,
    analyze_attack_ensemble
)

from .storage import (
    save_precompiled_distances,
    load_precompiled_distances,
    load_best_distances_with_wandb
)

__all__ = [
    # Main interface
    'get_stats',
    'compare_attacks',
    'compute_curves', 
    'compute_optimality',
    'compute_efficiency',
    
    # Component functions
    'compute_distance_statistics',
    'eval_optimality',
    'compute_basic_metrics',
    'compute_attack_efficiency',
    'compute_robust_accuracy_curve',
    'compute_auc_robust_accuracy', 
    'compute_certified_robustness_metrics',
    'ensemble_distances',
    'complementarity',
    'ensemble_gain',
    'analyze_attack_ensemble',
    'save_precompiled_distances',
    'load_precompiled_distances',
    'load_best_distances_with_wandb'
]