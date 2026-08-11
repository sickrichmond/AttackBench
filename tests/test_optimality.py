"""Analytic checks for the AUREC / local-optimality math (AttackBench Eq. 4 and 5)."""
import math

import pytest

from attackbench.metrics.distances import _aurec, compute_basic_metrics, eval_optimality

INF = float('inf')


def test_aurec_is_the_exact_area_of_the_step_function():
    # ref = [1,2,3,4] over 4 samples: rho drops .75/.5/.25/0 at each distance, so the
    # area over [0, 4] is 1*1 + .75 + .5 + .25 = 2.5
    assert math.isclose(_aurec([1, 2, 3, 4], 4.0, 4), 2.5)
    # attack = [2,3,4,inf]: 2*1 + .75 + .5 = 3.25
    assert math.isclose(_aurec([2, 3, 4, INF], 4.0, 4), 3.25)


def test_aurec_counts_unbroken_and_out_of_window_samples_as_robust():
    assert math.isclose(_aurec([INF] * 4, 4.0, 4), 4.0)
    assert math.isclose(_aurec([9, 9, 9, 9], 4.0, 4), 4.0)


def test_optimality_endpoints():
    ref = [1, 2, 3, 4]
    # identical to the lower envelope
    assert math.isclose(eval_optimality(ref, ref, clean_acc=1.0), 1.0)
    # (rho*eps0 - aurec_att) / (rho*eps0 - aurec_ref) = (4 - 3.25) / (4 - 2.5)
    assert math.isclose(eval_optimality([2, 3, 4, INF], ref, clean_acc=1.0), 0.5)
    # an attack that never succeeds
    assert math.isclose(eval_optimality([INF] * 4, ref, clean_acc=1.0), 0.0)
    # better than the reference saturates at 1.0 rather than exceeding it
    assert eval_optimality([0.5, 1, 1.5, 2], ref, clean_acc=1.0) == 1.0


def test_optimality_accounts_for_already_misclassified_samples():
    # a distance of 0 means the sample was already misclassified: clean accuracy is 0.75
    ref = [0, 1, 2, 3]
    assert math.isclose(eval_optimality(ref, ref, clean_acc=0.75), 1.0)


def test_unbroken_reference_warns_but_stays_finite():
    with pytest.warns(UserWarning, match='unbroken'):
        value = eval_optimality([1, 2, INF], [1, 2, INF], clean_acc=1.0)
    assert not math.isnan(value)


def test_degenerate_reference_is_nan_not_a_bogus_score():
    assert math.isnan(eval_optimality([1, 1], [1, 1], clean_acc=1.0))
    assert math.isnan(eval_optimality([1, 2], [], clean_acc=1.0))


def test_accuracy_is_not_the_error_rate():
    results = {'adv_success': [1, 1, 0, 1],
               'ori_success': [1, 0, 0, 0],   # one sample was already misclassified
               'correct': [0, 1, 1, 1]}
    metrics = compute_basic_metrics(results)
    assert math.isclose(metrics['accuracy'], 0.75)
    assert math.isclose(metrics['ASR'], 0.75)


def test_accuracy_falls_back_to_ori_success_for_older_results():
    metrics = compute_basic_metrics({'adv_success': [1], 'ori_success': [1, 0, 0, 0]})
    assert math.isclose(metrics['accuracy'], 0.75)
