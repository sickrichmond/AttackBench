Optimality Metrics
==================

AttackBenchLib introduces local and global optimality metrics for comparing adversarial attacks.

Overview
--------

Attack Success Rate at a single perturbation budget does not say how far an attack is
from the best achievable solution, and it depends on the budget you happen to pick.
AttackBench compares attacks over their whole *robustness evaluation curve* instead:

1. **Local optimality** (:math:`\xi^i_{\theta}`): how close attack :math:`a^i` gets to the
   best empirical solution on one model :math:`\theta`
2. **Global optimality** (:math:`\xi^i`): the average of local optimality over a set of
   models, which is what the leaderboard ranks

Robustness evaluation curve
---------------------------

For an attack :math:`a` on a model :math:`\theta`, the Attack Success Rate at a
perturbation budget :math:`\varepsilon` counts the samples the attack misclassifies
within that budget:

.. math::

   \text{ASR}_a(\varepsilon) = \frac{1}{|\mathcal{D}|}
   \sum_{(\mathbf{x}, y) \in \mathcal{D}} \mathbb{I}(d_{\mathbf{x}} \leq \varepsilon)

where :math:`d_{\mathbf{x}} = \lVert \mathbf{x}_{\text{adv}} - \mathbf{x} \rVert_p` is the
size of the perturbation the attack found for that sample. The **robust accuracy curve**
is its complement, :math:`\rho_a(\varepsilon) = 1 - \text{ASR}_a(\varepsilon)`, and the
area under it summarises the attack over every budget at once:

.. math::

   \text{AUREC}_a(\varepsilon_0) = \int_0^{\varepsilon_0} \rho_a(\varepsilon)\,d\varepsilon

A more effective attack pushes the curve towards the origin, so a *smaller* area is better.

Local Optimality
----------------

Definition
~~~~~~~~~~

Areas are not comparable across models, since each model starts from its own clean
accuracy. Local optimality normalises them against the best empirical attack
:math:`a^{\star}` — obtained by taking, for every sample, the smallest perturbation found
by *any* attack in the benchmark:

.. math::

   \xi^i_{\theta} = \frac{\rho \cdot \varepsilon_0 - \text{AUREC}_{a^i}(\varepsilon_0)}
                          {\rho \cdot \varepsilon_0 - \text{AUREC}_{a^{\star}}(\varepsilon_0)}

where :math:`\rho` is the model's clean accuracy and :math:`\varepsilon_0` is the smallest
budget at which the best attack drives robust accuracy to zero,
:math:`\rho_{a^{\star}}(\varepsilon_0) = 0`. The box :math:`\rho \cdot \varepsilon_0` is the
area of an attack that never succeeds, which bounds the score in :math:`[0, 1]`:

- :math:`\xi^i_{\theta} = 1` — the attack matches the best empirical solution on every sample
- :math:`\xi^i_{\theta} = 0` — the attack never evades the model within :math:`\varepsilon_0`

.. note::

   Distances are taken from ``results['distances']``, which holds :math:`d^{\star}`: the
   *smallest* perturbation found while the attack was running, not the sample the attack
   returned last. Attacks are also run under a fixed query budget (2000 forward+backward
   propagations per sample by default). Both are part of the protocol — see
   :doc:`architecture` — and both change the numbers, so scores are only comparable
   between runs that share them.

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

   # Compute local optimality for PGD using other attacks as reference (lower envelope)
   opt = attackbench.compute_local_optimality(
       attack_results=results_pgd,
       reference_results=[results_apgd, results_fab],
       threat_model='linf'
   )
   print(f"PGD optimality: {opt['optimality']:.3f}")

   # Or compare all attacks against each other
   comparison = attackbench.compare_attacks_optimality(
       [results_pgd, results_apgd, results_fab],
       threat_model='linf',
       attack_names=['PGD', 'APGD', 'FAB']
   )
   for name, score in comparison['ranking']:
       print(f"{name}: {score:.3f}")

When downloading optimal distances from W&B, sample matching is done via
SHA-512 hashes, not positional alignment. This means you can evaluate any
subset of the dataset and still get correct optimality values:

.. code-block:: python

   # Automatic: uses metadata from run_attack() and downloads optimal from W&B
   opt = attackbench.compute_local_optimality(results_apgd)
   print(f"APGD optimality: {opt['optimality']:.2%}")

   # get_stats also computes optimality consistently via compute_local_optimality
   stats = attackbench.get_stats(results_apgd, 'linf')
   print(f"Optimality from get_stats: {stats['optimality']:.2%}")
   # Both methods return the same result

.. warning::

   The reference :math:`a^{\star}` must come from *other* attacks — passed in explicitly
   or downloaded from W&B. If neither is available, ``get_stats()`` leaves the
   ``optimality`` key out and explains why, rather than falling back on the attack's own
   distances: comparing an attack with itself scores about 1.0 whatever it did.

   AttackBench also rejects W&B envelopes that lack the 2.x protocol and
   ``best_observed`` distance markers. Until a model/norm envelope has been repopulated
   from 2.x ``d*`` results, use explicit ``reference_results`` or expect automatic
   optimality to report that no compatible reference is available.

Global Optimality
-----------------

Definition
~~~~~~~~~~

Local optimality is measured against one model, so an attack tuned for that model can
score well without generalising. **Global optimality** averages it over a set of models
:math:`\mathcal{M} = \{\theta_1, \ldots, \theta_M\}`:

.. math::

   \xi^i = \frac{1}{|\mathcal{M}|} \sum_{\theta_m \in \mathcal{M}} \xi^i_{\theta_m}

It is bounded in :math:`[0, 1]` like local optimality, and equals 1 only for an attack
that finds the minimal perturbation of every sample on every model in the set.

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

   # A leaderboard is built from the raw results of several attacks, not from a
   # global-optimality result: attack_name -> model_name -> results of run_attack()
   leaderboard = attackbench.create_attack_leaderboard(
       {
           'PGD': pgd_results_per_model,
           'APGD': apgd_results_per_model,
           'FMN': fmn_results_per_model,
       },
       threat_model='linf'
   )
   print(attackbench.format_leaderboard(leaderboard))

Attack Leaderboard
------------------

The leaderboard ranks attacks by their global optimality score, grouped by the
:math:`\ell_p` threat model they assume (illustrative shape, not measured values):

.. code-block:: text

   Rank  Attack    Global Optimality
   ----  ------    -----------------
   1     ...       0.9xx
   2     ...       0.9xx
   3     ...       0.8xx

For the published rankings see the leaderboard at https://attackbench.github.io/ and
Table I of the paper.

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

   # BoMN is the lower envelope of the attacks you gave it, so it scores
   # LocalOpt = 1.0 against *those* attacks. Against the W&B envelope — computed
   # over every attack in the benchmark — it scores below 1.0 like anything else.

BoMN represents an upper bound on attack performance given the set of attacks used.

Ensemble Gain
--------------

Measure what a second attack adds to a first one — the fraction of samples it breaks
and the other does not:

.. code-block:: python

   import numpy as np

   gain = attackbench.ensemble_gain(
       np.array(results_pgd['adv_success']),
       np.array(results_apgd['adv_success']),
   )

   print(f"Samples APGD breaks that PGD does not: {gain:.2%}")

Comparing Attacks
-----------------

Compare multiple attacks with a comprehensive summary:

.. code-block:: python

   comparison = attackbench.compare_attacks(
       [results_pgd, results_apgd, results_fab],
       threat_model='linf'
   )

   # Compare optimality across attacks: a list of results plus their names
   opt_comparison = attackbench.compare_attacks_optimality(
       [results_pgd, results_apgd, results_fab],
       threat_model='linf',
       attack_names=['PGD', 'APGD', 'FAB']
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
