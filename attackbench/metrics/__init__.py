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
- compute_local_optimality(): Stage 3 - Local optimality (user-friendly API)
- compare_attacks_optimality(): Compare multiple attacks' optimality
"""

# Main analysis functions
from .analysis import (
    get_stats,
    compare_attacks,
    compute_curves,
    compute_optimality,
    compute_efficiency
)

# Stage 3: Optimality (user-friendly API)
from .optimality import (
    compute_local_optimality,
    compare_attacks_optimality
)

# Stage 4-5: Global Optimality & Ranking (user-friendly API)
from .global_optimality import (
    compute_global_optimality,
    create_attack_leaderboard,
    compare_attacks_global,
    format_leaderboard
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
    load_precompiled_distances
)

__all__ = [
    # Main interface
    'get_stats',
    'compare_attacks',
    'compute_curves',
    'compute_optimality',
    'compute_efficiency',

    # Optimality (stages 3-5)
    'compute_local_optimality',
    'compare_attacks_optimality',
    'compute_global_optimality',
    'create_attack_leaderboard',
    'compare_attacks_global',
    'format_leaderboard',

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
    'load_precompiled_distances'
]
