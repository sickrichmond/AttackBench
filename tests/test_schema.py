"""BoMN composes attacks and its output feeds the analysis pipeline unchanged."""
import math

import pytest
import torch

from attackbench.attacks.bomn import bomn_attack
from attackbench.metrics.analysis import get_stats
from attackbench import run_attack
from conftest import lazy_attack, loose_attack, tight_attack

CPU = torch.device('cpu')


def test_bomn_keeps_the_best_attack_per_sample(model, loader):
    tight = run_attack(model, loader, tight_attack, 'linf', device=CPU)
    composite = bomn_attack(model, loader, [loose_attack, tight_attack, lazy_attack],
                            'linf', device=CPU, verbose=False)

    # tight_attack wins everywhere, so the envelope is exactly its distances
    assert composite['distances']['linf'] == tight['distances']['linf']
    assert composite['best_attack_indices'] == [1] * len(tight['distances']['linf'])
    assert composite['attack_names'][1].startswith('tight_attack')

    # the composite spends what its components spent
    assert sum(composite['num_forwards']) >= sum(tight['num_forwards'])


def test_bomn_output_is_a_run_attack_result(model, loader):
    composite = bomn_attack(model, loader, [tight_attack, loose_attack], 'linf',
                            device=CPU, verbose=False)
    single = run_attack(model, loader, tight_attack, 'linf', device=CPU)

    assert set(single).issubset(set(composite))
    assert composite['hashes'] == single['hashes']

    # get_stats used to raise AttributeError on BoMN results
    stats = get_stats(composite, 'linf', include_optimality=False)
    assert math.isclose(stats['accuracy'], 1.0)
    assert math.isclose(stats['ASR'], 1.0)


def test_bomn_all_failed_marks_no_winner(model, loader):
    composite = bomn_attack(model, loader, [lazy_attack, lazy_attack], 'linf',
                            device=CPU, verbose=False)
    assert all(math.isinf(d) for d in composite['distances']['linf'])
    assert composite['best_attack_indices'] == [-1] * len(composite['hashes'])


def test_get_stats_needs_a_real_reference_for_optimality(model, loader):
    results = run_attack(model, loader, loose_attack, 'linf', device=CPU)

    # no reference available and no network: optimality is skipped, not faked
    stats = get_stats(results, 'linf', auto_load_best=False)
    assert 'optimality' not in stats

    # against a stricter reference the attack scores below 1.0
    better = run_attack(model, loader, tight_attack, 'linf', device=CPU)
    stats = get_stats(results, 'linf', auto_load_best=False,
                      best_distances=better['distances']['linf'])
    assert 0.0 <= stats['optimality'] < 1.0
    assert stats['optimality_reference'] == 'ensemble_1'


def test_precompiled_schema_rejects_legacy_artifacts():
    from attackbench.wandb.manager import _make_artifact_name, _validate_precompiled_data

    assert _make_artifact_name("CIFAR10", "L2", "Standard") == "cifar10-l2-standard"
    legacy = {"distances": {"linf": [0.1]}}
    with pytest.warns(UserWarning, match="incompatible pre-2.0"):
        assert _validate_precompiled_data(legacy, "legacy-result") is None


def test_precompiled_schema_accepts_complete_results():
    from attackbench.wandb.manager import (
        PRECOMPILED_RESULT_FIELDS,
        _validate_precompiled_data,
    )

    complete = {field: [] for field in PRECOMPILED_RESULT_FIELDS}
    complete["query_budget"] = 2000

    assert _validate_precompiled_data(complete, "current-result") is complete


def test_optimal_schema_rejects_pre_2_distance_semantics():
    from attackbench.wandb.manager import _validate_optimal_data

    legacy = {
        "distances": {"linf": {"hash": 0.1}},
        "metadata": {"format": "hash_based"},
    }
    with pytest.warns(UserWarning, match="incompatible pre-2.0"):
        assert _validate_optimal_data(legacy, "legacy-envelope") is None


def test_optimal_schema_accepts_current_lower_envelopes():
    from attackbench.wandb.manager import (
        DISTANCE_SEMANTICS,
        PROTOCOL_VERSION,
        _validate_optimal_data,
    )

    current = {
        "distances": {"linf": {"hash": 0.1}},
        "metadata": {
            "format": "hash_based",
            "protocol_version": PROTOCOL_VERSION,
            "distance_semantics": DISTANCE_SEMANTICS,
        },
    }

    assert _validate_optimal_data(current, "current-envelope") is current


def test_legacy_results_cannot_update_current_envelope():
    from attackbench.wandb.manager import update_optimal_distances

    legacy = {
        "hashes": ["hash"],
        "distances": {"linf": [0.1]},
        "metadata": {
            "dataset": "toy",
            "model_name": "model",
            "threat_model": "linf",
        },
    }

    with pytest.raises(ValueError, match="legacy results cannot update"):
        update_optimal_distances(legacy, dry_run=True)
