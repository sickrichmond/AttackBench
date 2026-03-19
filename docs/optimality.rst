Optimality Metrics
==================

AttackBenchLib introduces local and global optimality metrics for comparing adversarial attacks.

Overview
--------

Traditional metrics like Attack Success Rate (ASR) don't capture how optimal the generated
adversarial examples are. AttackBenchLib addresses this by:

1. **Local Optimality**: Comparing attacks on individual model-sample pairs
2. **Global Optimality**: Aggregating optimality across models and samples

Local Optimality
----------------

Definition
~~~~~~~~~~

For a given model and sample, the **local optimality** of an attack measures how close its
perturbation is to the minimum possible perturbation.

Given multiple attacks :math:`A_1, A_2, \ldots, A_n` and their generated perturbation
distances :math:`d_1, d_2, \ldots, d_n`, the best (minimum) distance is:

.. math::

   d_{\text{best}} = \min(d_1, d_2, \ldots, d_n)

The local optimality of attack :math:`A_i` is:

.. math::

   \text{LocalOpt}(A_i) = \frac{d_{\text{best}}}{d_i}

where:

- :math:`\text{LocalOpt}(A_i) = 1.0` indicates optimal (smallest) perturbation
- :math:`\text{LocalOpt}(A_i) < 1.0` indicates sub-optimal perturbation

Computing Local Optimality
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local optimality can be computed in two ways:

1. **From multiple attack results** (relative comparison):
   The best distance per sample is determined element-wise across all provided results.

2. **From precomputed optimal distances** (absolute comparison):
   Optimal distances stored on W&B (hash-based lookup tables computed over the
   full dataset with all available attacks) are matched to each sample by its
   SHA-512 hash. This is the recommended approach as it gives a stable reference
   independent of which attacks or subset are used.

.. code-block:: python

   import attackbench
   from attackbench.attacks import pgd, apgd, fab

   # Run multiple attacks
   results_pgd = attackbench.run_attack(model, dataset, pgd, 'linf', device)
   results_apgd = attackbench.run_attack(model, dataset, apgd, 'linf', device)
   results_fab = attackbench.run_attack(model, dataset, fab, 'linf', device)

   # Compute local optimality (requires attackbench[metrics])
   optimality = attackbench.compute_local_optimality(
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

When ``optimal_distances`` (a hash-based dict ``{sha512_hash: distance}``) are
provided, the function matches each sample via its hash rather than relying on
positional alignment. This means you can evaluate any subset of the dataset and
still get correct optimality values:

.. code-block:: python

   # Download hash-based optimal distances from W&B
   optimal = attackbench.download_optimal_distances(
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard'
   )

   optimality = attackbench.compute_local_optimality(
       attack_results={'APGD': results_apgd},
       threat_model='linf',
       optimal_distances=optimal   # hash-based lookup
   )

Global Optimality
-----------------

Definition
~~~~~~~~~~

**Global optimality** aggregates local optimality across multiple models to provide
an overall ranking of attacks.

For attack :math:`A_i` evaluated on :math:`M` models:

.. math::

   \text{GlobalOpt}(A_i) = \frac{1}{M} \sum_{m=1}^{M} \overline{\text{LocalOpt}_m}(A_i)

where :math:`\overline{\text{LocalOpt}_m}(A_i)` is the mean local optimality on model :math:`m`.

Computing Global Optimality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Dictionary mapping model_name -> attack results from run_attack()
   results_per_model = {}
   for model_name in ['Standard', 'Carmon2019Unlabeled', 'Wong2020Fast']:
       model = attackbench.load_model(model_name, 'cifar10', 'Linf')
       results = attackbench.run_attack(model, dataset, pgd, 'linf', device)
       results_per_model[model_name] = results

   # Compute global optimality
   # When no reference is provided, optimal distances are downloaded from W&B
   global_opt = attackbench.compute_global_optimality(
       results_per_model,
       threat_model='linf'
   )

   # You can also provide custom reference results per model
   # (single dicts are automatically wrapped in lists)
   global_opt = attackbench.compute_global_optimality(
       results_per_model,
       threat_model='linf',
       reference_per_model={
           'Standard': [results_apgd_std, results_fab_std],
           'Carmon2019Unlabeled': results_apgd_carmon,  # single dict OK
       },
       use_wandb=False  # don't download from W&B
   )

   # Create and display leaderboard
   leaderboard = attackbench.create_attack_leaderboard(global_opt)
   print(attackbench.format_leaderboard(leaderboard))

Attack Leaderboard
------------------

The leaderboard ranks attacks by their global optimality score:

.. code-block:: text

   Rank  Attack    Global Optimality
   ----  ------    -----------------
   1     FAB-T     0.987
   2     APGD      0.964
   3     PGD       0.845
   4     FGSM      0.623

The official leaderboard is available at: https://attackbench.github.io/

Best-of-MinNorm (BoMN)
----------------------

BoMN creates an ensemble by selecting the best attack result per sample:

.. code-block:: python

   results_bomn = attackbench.bomn_attack(
       model=model,
       dataset=dataset,
       attacks=[pgd, apgd, fab],
       threat_model='linf',
       device=device
   )

   # BoMN achieves the best possible performance
   # by definition: LocalOpt = 1.0 for all samples

BoMN represents an upper bound on attack performance given the set of attacks used.

Ensemble Gain
--------------

Measure the improvement from using multiple attacks together:

.. code-block:: python

   gain = attackbench.ensemble_gain(
       individual_results=[results_pgd, results_apgd, results_fab],
       bomn_results=results_bomn,
       threat_model='linf'
   )

   print(f"Ensemble gain over best individual: {gain:.2%}")

Comparing Attacks
-----------------

Compare multiple attacks with a comprehensive summary:

.. code-block:: python

   comparison = attackbench.compare_attacks(
       [results_pgd, results_apgd, results_fab],
       threat_model='linf'
   )

   # Compare optimality across attacks
   opt_comparison = attackbench.compare_attacks_optimality(
       attack_results={
           'PGD': results_pgd,
           'APGD': results_apgd,
           'FAB': results_fab
       },
       threat_model='linf'
   )

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

  Cina, A. E., Rony, J., Pintor, M., et al. (2025).
  *AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples*.
  Proceedings of the AAAI Conference on Artificial Intelligence.
