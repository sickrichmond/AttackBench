Optimality Metrics
==================

AttackBench introduces local and global optimality metrics for comparing adversarial attacks.

Overview
--------

Traditional metrics like Attack Success Rate (ASR) don't capture how optimal the generated adversarial examples are. AttackBench addresses this by:

1. **Local Optimality**: Comparing attacks on individual model-sample pairs
2. **Global Optimality**: Aggregating optimality across models and samples

Local Optimality
----------------

Definition
~~~~~~~~~~

For a given model and sample, the **local optimality** of an attack measures how close its perturbation is to the minimum possible perturbation.

Given multiple attacks :math:`A_1, A_2, ..., A_n` and their generated perturbation distances :math:`d_1, d_2, ..., d_n`, the best (minimum) distance is:

.. math::

   d_{best} = \\min(d_1, d_2, ..., d_n)

The local optimality of attack :math:`A_i` is:

.. math::

   \\text{LocalOpt}(A_i) = \\frac{d_{best}}{d_i}

where:

- :math:`\\text{LocalOpt}(A_i) = 1.0` indicates optimal (smallest) perturbation
- :math:`\\text{LocalOpt}(A_i) < 1.0` indicates sub-optimal perturbation

Computing Local Optimality
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from attackbench import compute_local_optimality

   # Run multiple attacks
   results_pgd = run_attack(model, dataset, pgd, 'linf', device)
   results_apgd = run_attack(model, dataset, apgd, 'linf', device)
   results_fab = run_attack(model, dataset, fab, 'linf', device)

   # Compute local optimality
   optimality = compute_local_optimality(
       attack_results={
           'PGD': results_pgd,
           'APGD': results_apgd,
           'FAB': results_fab
       },
       threat_model='linf'
   )

   # Results per attack
   for attack_name, opt_values in optimality.items():
       mean_opt = sum(opt_values) / len(opt_values)
       print(f"{attack_name}: {mean_opt:.3f}")

Global Optimality
-----------------

Definition
~~~~~~~~~~

**Global optimality** aggregates local optimality across multiple models to provide an overall ranking of attacks.

For attack :math:`A_i` evaluated on :math:`M` models:

.. math::

   \\text{GlobalOpt}(A_i) = \\frac{1}{M} \\sum_{m=1}^{M} \\overline{\\text{LocalOpt}_m}(A_i)

where :math:`\\overline{\\text{LocalOpt}_m}(A_i)` is the mean local optimality on model :math:`m`.

Computing Global Optimality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from attackbench import compute_global_optimality, create_attack_leaderboard

   # Dictionary of {model_name: {attack_name: results}}
   all_results = {
       'Standard': {
           'PGD': results_pgd_std,
           'APGD': results_apgd_std,
       },
       'Carmon2019': {
           'PGD': results_pgd_carmon,
           'APGD': results_apgd_carmon,
       }
   }

   # Compute global optimality
   global_opt = compute_global_optimality(
       all_results,
       threat_model='linf'
   )

   # Create leaderboard
   leaderboard = create_attack_leaderboard(global_opt)
   print(leaderboard)

Attack Leaderboard
------------------

The leaderboard ranks attacks by their global optimality score:

.. code-block:: text

   Rank  Attack    Global Optimality  Avg ASR
   ----  ------    -----------------  -------
   1     FAB-T     0.987             98.5%
   2     APGD      0.964             97.2%
   3     PGD       0.845             95.1%
   4     FGSM      0.623             78.3%

Best-of-MinNorm (BoMN)
----------------------

BoMN creates an ensemble by selecting the best attack result per sample:

.. code-block:: python

   from attackbench import bomn_attack

   results_bomn = bomn_attack(
       model=model,
       dataset=dataset,
       attacks=[pgd, apgd, fab],
       threat_model='linf',
       device=device
   )

   # BoMN achieves the best possible performance
   # by definition: LocalOpt = 1.0 for all samples

BoMN represents an upper bound on attack performance given the set of attacks used.

Robust Accuracy Curves
----------------------

Visualize attack performance across perturbation budgets:

.. code-block:: python

   from attackbench import plot_robust_accuracy_curve

   plot_robust_accuracy_curve(
       results_dict={
           'PGD': results_pgd,
           'APGD': results_apgd,
           'FAB': results_fab
       },
       threat_model='linf',
       save_path='robustness_curve.pdf'
   )

The curve shows model accuracy vs. perturbation size, allowing comparison of attack strength.

Computing Ensemble Gain
------------------------

Measure the improvement from using multiple attacks:

.. code-block:: python

   from attackbench import ensemble_gain

   gain = ensemble_gain(
       individual_results=[results_pgd, results_apgd, results_fab],
       bomn_results=results_bomn,
       threat_model='linf'
   )

   print(f"Ensemble gain over best individual: {gain:.2%}")

Use Cases
---------

**Attack Development**
  Measure how your new attack compares to existing methods

**Library Selection**
  Choose which attack library implementation to use

**Model Evaluation**
  Assess model robustness using the strongest possible attacks

**Research Benchmarking**
  Provide reproducible, fair attack comparisons

References
----------

For more details on the optimality metrics, see:

  Cinà, A. E., Rony, J., Pintor, M., et al. (2025). 
  *AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples*. 
  Proceedings of the AAAI Conference on Artificial Intelligence.
