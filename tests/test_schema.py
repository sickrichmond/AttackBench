"""BoMN composes attacks and its output feeds the analysis pipeline unchanged."""
import math

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
