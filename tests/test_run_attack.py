"""run_attack end to end on CPU: d*, clean accuracy, hashes and the query budget."""
import math

import pytest
import torch

from attackbench import run_attack
from conftest import (discarding_attack, lazy_attack, loose_attack, tight_attack,
                      wasteful_attack, make_loader)


def test_runs_on_cpu_and_reports_the_expected_distances(model, loader):
    results = run_attack(model, loader, tight_attack, 'linf', device=torch.device('cpu'))

    # every sample is broken, at exactly the distance the attack moved the pixel
    assert all(results['adv_success'])
    expected = [abs(x[0, 0, 0].item() - (0.5 - 0.01 if x[0, 0, 0] > 0.5 else 0.5 + 0.01))
                for x, _ in loader.dataset]
    for got, want in zip(results['distances']['linf'], expected):
        assert math.isclose(got, want, rel_tol=1e-5), (got, want)

    # the model is right on every clean sample, and accuracy is not the error rate
    assert results['correct'] == [True] * len(expected)
    assert results['ori_success'] == [False] * len(expected)


def test_distances_are_the_best_iterate_not_the_last(model, loader):
    """The headline fix: an attack that discards its own best result must not be
    credited with the distance of what it returned."""
    results = run_attack(model, loader, discarding_attack, 'linf', device=torch.device('cpu'))

    best = results['distances']['linf']
    final = results['final_distances']['linf']

    def distance_to_boundary(v, margin):
        return abs(v - (0.5 - margin if v > 0.5 else 0.5 + margin))

    pixels = [x[0, 0, 0].item() for x, _ in loader.dataset]
    # d* must come from the 0.01 probe the attack queried, not from what it returned
    for got, v in zip(best, pixels):
        assert math.isclose(got, distance_to_boundary(v, 0.01), rel_tol=1e-5)
    for got, v in zip(final, pixels):
        assert math.isclose(got, distance_to_boundary(v, 0.4), rel_tol=1e-5)
    assert all(b < f for b, f in zip(best, final))


def test_failed_attack_gets_infinite_distance(model, loader):
    results = run_attack(model, loader, lazy_attack, 'linf', device=torch.device('cpu'))

    assert not any(results['adv_success'])
    assert all(math.isinf(d) for d in results['distances']['linf'])
    assert all(math.isinf(d) for d in results['final_distances']['linf'])


def test_query_budget_is_enforced(model, loader):
    budget = 10
    results = run_attack(model, loader, wasteful_attack, 'linf',
                         device=torch.device('cpu'), query_budget=budget)

    spent = [f + b for f, b in zip(results['num_forwards'], results['num_backwards'])]
    assert max(spent) <= budget, spent
    assert results['query_budget'] == budget

    # without a budget the same attack is free to burn every query it asks for
    unbudgeted = run_attack(model, loader, wasteful_attack, 'linf',
                            device=torch.device('cpu'), query_budget=None)
    assert max(unbudgeted['num_forwards']) > budget


def test_hashes_identify_the_samples(model, loader):
    a = run_attack(model, loader, tight_attack, 'linf', device=torch.device('cpu'))
    b = run_attack(model, loader, loose_attack, 'linf', device=torch.device('cpu'))
    assert a['hashes'] == b['hashes']
    assert len(set(a['hashes'])) == len(a['hashes'])

    other = run_attack(model, make_loader(n=8, batch_size=2), tight_attack, 'linf',
                       device=torch.device('cpu'))
    assert other['hashes'] == a['hashes']  # batching must not change sample identity


def test_diagnostics_are_always_present(model, loader):
    results = run_attack(model, loader, tight_attack, 'linf', device=torch.device('cpu'))
    for key in ('num_forwards', 'num_backwards', 'times', 'box_failures',
                'batch_failures', 'correct', 'original_predictions',
                'adversarial_predictions', 'hashes'):
        assert key in results, key
    assert not any(results['batch_failures'])
    assert not any(results['box_failures'])


def test_failing_attack_is_recorded_not_raised(model, loader):
    def exploding_attack(model, inputs, labels, **kwargs):
        raise RuntimeError('boom')

    with pytest.warns(UserWarning):
        results = run_attack(model, loader, exploding_attack, 'linf',
                             device=torch.device('cpu'))

    assert all(results['batch_failures'])
    assert not any(results['adv_success'])

    with pytest.raises(RuntimeError):
        run_attack(model, loader, exploding_attack, 'linf',
                   device=torch.device('cpu'), debug=True)
