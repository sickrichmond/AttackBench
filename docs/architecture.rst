Architecture
============

AttackBenchLib is organized into several key modules that work together to provide a comprehensive
adversarial attack benchmarking framework.

Overview
--------

The framework follows the five stages of the AttackBench paper:

1. **Model Zoo**: a diverse pool of robust and non-robust models
2. **Attack Benchmarking**: every attack runs against every model under the same query
   budget, recording the best perturbation found rather than the last iterate
3. **Local Optimality**: how close each attack gets to the best empirical attack on a
   given model
4. **Global Optimality**: the average of local optimality across the model zoo
5. **Ranking**: attacks ranked by global optimality, grouped by threat model

Package Structure
-----------------

.. code-block:: text

   attackbench/
   ├── __init__.py              # Public API with lazy imports
   ├── run.py                   # Core run_attack() function
   ├── custom_components.py     # create_custom_attack()
   ├── preconfigured.py         # Ready-made attacks (pgd, fgsm, apgd, etc.)
   ├── adv_lib_sub.py           # Internal distance metrics and utilities
   ├── attacks/                 # Attack wrappers and registry system
   │   ├── registry.py          # get_attack(), list_attacks() - dynamic attack loading
   │   ├── bomn.py              # Best-of-MinNorm composite attack
   │   ├── original/            # Native AttackBenchLib implementations
   │   ├── art/                 # IBM ART library wrappers
   │   ├── foolbox/             # Foolbox library wrappers
   │   ├── torchattacks/        # Torchattacks library wrappers
   │   ├── adv_lib/             # Adversarial Library wrappers
   │   ├── cleverhans/          # CleverHans wrappers
   │   └── deeprobust/          # DeepRobust wrappers
   ├── datasets/                # Dataset loading
   │   └── registry.py          # get_loader(), get_dataset()
   ├── models/                  # Model loading
   │   ├── registry.py          # get_model()
   │   ├── benchmodel_wrapper.py  # BenchModel wrapper
   │   └── original/            # Custom model implementations
   ├── metrics/                 # Analysis and evaluation
   │   ├── analysis.py          # get_stats(), compare_attacks()
   │   ├── curves.py            # Robust accuracy curves and AUC
   │   ├── distances.py         # Distance computation and eval_optimality
   │   ├── ensemble.py          # Ensemble gain metrics
   │   ├── optimality.py        # Local optimality computation
   │   ├── global_optimality.py # Global optimality and leaderboard
   │   └── storage.py           # Results storage utilities
   └── wandb/                   # W&B integration for results sharing
                                   # (precompiled & optimal distances)

   scripts/
   ├── paper_acceptance.py      # Run attacks under the paper's protocol and report
   └── check_config_equivalence.py
   tests/                       # CPU test suite (pytest)

Core Modules
------------

attackbench (top-level)
~~~~~~~~~~~~~~~~~~~~~~~

The main package provides the high-level API through ``__init__.py``. It uses
PEP 562 lazy imports to keep the base installation lightweight:

- **Always available** (core):
  - ``run_attack()``: Execute attacks and collect raw results
  - ``create_custom_attack()``: Wrap user-defined attack functions
  - ``get_loader()``: Load a dataset as a DataLoader
  - ``DEFAULT_QUERY_BUDGET``: the protocol's budget (2000)

- **Requires** ``attackbenchlib[models]``:
  - ``get_model()``: Load a model from the model registry
  - ``load_model()``: Load a RobustBench model with metadata

- **Requires** ``attackbenchlib[attacks]``:
  - ``get_attack()``: Instantiate ART, Foolbox, CleverHans, and original attacks
  - ``list_attacks()``: Discover available attacks
  - ``bomn_attack()``: Best-of-MinNorm composite attack

- **Requires** ``attackbenchlib[torchattacks]``:
  - Torchattacks wrappers, isolated because upstream pins ``requests~=2.25.1``

- **Requires** ``attackbenchlib[metrics]``:
  - ``get_stats()``: Compute statistics from attack results
  - ``compute_local_optimality()``, ``compute_global_optimality()``: Optimality metrics
  - ``create_attack_leaderboard()``, ``format_leaderboard()``: Ranking utilities
  - ``compare_attacks()``, ``ensemble_gain()``: Multi-attack comparison

- **W&B integration** (always available, requires wandb):
  - ``upload_precompiled_distances()``, ``download_precompiled_distances()``
  - ``upload_optimal_distances()``, ``download_optimal_distances()``
  - ``update_optimal_distances()``: fold a run into the published lower envelope (stage 5)

run.py
~~~~~~

Contains the core ``run_attack()`` function. Key features:

- **Automatic metadata extraction**: When objects are created via ``get_model()``,
  ``get_loader()``, and ``get_attack()``, metadata (dataset name, model name,
  attack name/library) is automatically attached and extracted.
- **BenchModel wrapping**: Models are automatically wrapped in ``BenchModel`` for
  query tracking and constraint enforcement.
- **Query budget**: every attack is limited to ``query_budget`` forward+backward
  propagations per sample (default 2000, the budget used in the paper). This is what
  makes attacks comparable; pass ``query_budget=None`` to lift it.
- **W&B caching**: With ``use_cached=True`` (off by default), checks W&B for existing
  precompiled distances and reuses only complete 2.x results whose query budget and
  per-sample hashes match.
- **SHA-512 hashing**: Computes a per-image hash on raw RGB values so that
  each sample is uniquely identifiable regardless of subset ordering.
- **Raw output**: Returns per-sample data only — distances, success flags, hashes,
  query counts and failure indicators. Use ``get_stats()`` for the statistics.
- **Untargeted**: like the paper's evaluation, attacks are run against the untargeted
  objective; the ``targets``/``targeted`` arguments exist for attacks that require them
  in their signature.

custom_components.py
~~~~~~~~~~~~~~~~~~~~

Utilities for integrating user-defined attacks:

- ``create_custom_attack()``: Wraps a user function with input validation,
  output processing, and constraint enforcement (e.g., clamping to [0, 1]).

preconfigured.py
~~~~~~~~~~~~~~~~

Preconfigured callables for immediate use:

- ``pgd``, ``fgsm``, ``apgd``, ``fab``, ``fmn``, ``deepfool``,
  ``superdeepfool``, ``trust_region``

These are created from the ``original/`` implementations and include
``_attackbench_name`` and ``_attackbench_lib`` metadata for automatic tracking.
The callables are imported lazily. FMN loads Foolbox and EagerPy when invoked and requires
the ``attacks`` extra; the remaining implementations use core dependencies.

BenchModel Wrapper
~~~~~~~~~~~~~~~~~~

The ``BenchModel`` class (in ``models/benchmodel_wrapper.py``) wraps any
``nn.Module`` to provide:

- Forward/backward query counting
- Timing measurements
- Minimum distance tracking per norm
- Box constraint enforcement ([0, 1] input range)
- Query budget limiting

Attack Format
-------------

All attack implementations follow a standardized interface:

**Inputs:**

- ``model``: ``nn.Module`` taking inputs in [0, 1] and returning logits
- ``inputs``: ``FloatTensor`` representing input samples in [0, 1]
- ``labels``: ``LongTensor`` representing true labels
- ``targets``: ``LongTensor`` or ``None`` for targeted attacks (``run_attack()``
  evaluates the untargeted objective, so it passes ``None``/``False``)
- ``targeted``: ``bool`` flag for targeted mode

**Output:**

- ``adv_inputs``: ``FloatTensor`` of perturbed inputs in [0, 1]

This standardization ensures fair comparison across different implementations.
The ``_call_attack()`` helper in ``run.py`` automatically filters kwargs to match
the attack's signature, so attacks only need to accept the parameters they use.

Results Format
--------------

The ``run_attack()`` function returns a dictionary with minimal raw data:

**Essential data (always returned):**

- ``distances``: Dict mapping norm names (``'l0'``, ``'l1'``, ``'l2'``, ``'linf'``)
  to lists of ``d*`` — the *smallest* perturbation found during the optimization, as
  defined in Algorithm 1 of the AttackBench paper. This is what optimality is computed
  on. ``0`` for already-misclassified samples, ``inf`` for samples never misclassified.
- ``final_distances``: Distance of the sample the attack actually returned (its last
  iterate). Always ``>= distances``; a large gap means the attack throws away its own
  best result. Diagnostics only.
- ``adv_success``: Boolean list indicating successful adversarial examples
- ``ori_success``: Boolean list indicating samples that were ALREADY misclassified
  before the attack
- ``correct``: Boolean list of clean correctness — ``accuracy`` is ``mean(correct)``
- ``hashes``: SHA-512 hashes of each input image (computed on raw RGB values).
  Always included to enable hash-based matching with optimal distances.

**Diagnostics (also always returned):**

- ``original_predictions``: Model predictions on clean inputs
- ``adversarial_predictions``: Model predictions on adversarial inputs
- ``num_forwards``: Forward pass count per sample
- ``num_backwards``: Backward pass count per sample
- ``times``: Execution time per batch
- ``box_failures``: The attack produced values outside ``[0, 1]`` (they get clipped)
- ``batch_failures``: The attack raised an exception on that batch
- ``query_budget``: The budget the run was executed under

**Optional tensors** (when ``save_adversarial=True``):

- ``adv_inputs``: Adversarial examples tensor
- ``inputs``: Original inputs tensor

For all statistics and analysis, pass results to ``get_stats()``.

Extension Points
----------------

Custom Attacks
~~~~~~~~~~~~~~

Use ``create_custom_attack()`` for validated wrappers:

.. code-block:: python

   def my_attack(model, inputs, labels, eps=0.3):
       # Your attack logic
       return adv_inputs

   attack = attackbench.create_custom_attack(my_attack, attack_name="MyAttack")

Or use attacks directly as callables:

.. code-block:: python

   results = attackbench.run_attack(model, dataset, my_attack, 'linf', device)

Custom Datasets
~~~~~~~~~~~~~~~

Create a standard PyTorch ``DataLoader`` and pass it to ``run_attack()``:

.. code-block:: python

   from torch.utils.data import DataLoader

   loader = DataLoader(my_dataset, batch_size=128, shuffle=False)
   results = attackbench.run_attack(model, loader, attack, 'linf', device)

Custom Models
~~~~~~~~~~~~~

Any ``nn.Module`` that accepts inputs in [0, 1] and returns logits works:

.. code-block:: python

   results = attackbench.run_attack(my_model, dataset, attack, 'linf', device)
