Architecture
============

AttackBench is organized into several key modules that work together to provide a comprehensive
adversarial attack benchmarking framework.

Overview
--------

The framework follows a five-stage evaluation process:

1. **Model Selection**: Select diverse robust and non-robust models
2. **Attack Execution**: Run attacks in a standardized environment
3. **Local Optimality**: Compute local optimality metrics
4. **Results Aggregation**: Aggregate results across models
5. **Global Ranking**: Rank attacks by global optimality

Package Structure
-----------------

.. code-block:: text

   attackbench/
   ├── __init__.py              # Public API with lazy imports
   ├── run.py                   # Core run_attack() function
   ├── custom_components.py     # create_custom_attack(), create_iterative_attack()
   ├── preconfigured.py         # Pre-instantiated attacks (pgd, fgsm, apgd, etc.)
   ├── compat.py                # NumPy compatibility layer
   ├── adv_lib_sub.py           # Internal distance metrics and utilities
   ├── attacks/                 # Attack wrappers and registry system
   │   ├── registry.py          # get_attack(), list_attacks() - dynamic attack loading
   │   ├── bomn.py              # Best-of-MinNorm composite attack
   │   ├── original/            # Native AttackBench implementations
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

Core Modules
------------

attackbench (top-level)
~~~~~~~~~~~~~~~~~~~~~~~

The main package provides the high-level API through ``__init__.py``. It uses
PEP 562 lazy imports to keep the base installation lightweight:

- **Always available** (core):
  - ``run_attack()``: Execute attacks and collect raw results
  - ``create_custom_attack()``: Wrap user-defined attack functions
  - ``get_model()``: Load a model from the model registry
  - ``get_loader()``: Load a dataset as a DataLoader
  - ``load_model()``: Load a RobustBench model with metadata

- **Requires** ``attackbench[attacks]``:
  - ``get_attack()``: Instantiate library attacks dynamically
  - ``list_attacks()``: Discover available attacks
  - ``bomn_attack()``: Best-of-MinNorm composite attack

- **Requires** ``attackbench[metrics]``:
  - ``get_stats()``: Compute statistics from attack results
  - ``compute_local_optimality()``, ``compute_global_optimality()``: Optimality metrics
  - ``create_attack_leaderboard()``, ``format_leaderboard()``: Ranking utilities
  - ``compare_attacks()``, ``ensemble_gain()``: Multi-attack comparison

- **W&B integration** (always available, requires wandb):
  - ``upload_precompiled_distances()``, ``download_precompiled_distances()``
  - ``list_available_distances()``

run.py
~~~~~~

Contains the core ``run_attack()`` function. Key features:

- **Automatic metadata extraction**: When objects are created via ``get_model()``,
  ``get_loader()``, and ``get_attack()``, metadata (dataset name, model name,
  attack name/library) is automatically attached and extracted.
- **BenchModel wrapping**: Models are automatically wrapped in ``BenchModel`` for
  query tracking and constraint enforcement.
- **W&B caching**: If ``use_cached=True`` (default), checks W&B for existing
  precompiled distances before running the attack.
- **Minimal output**: Returns only essential data (distances, success flags).
  Use ``get_stats()`` for analysis.

custom_components.py
~~~~~~~~~~~~~~~~~~~~

Utilities for integrating user-defined attacks:

- ``create_custom_attack()``: Wraps a user function with input validation,
  output processing, and constraint enforcement (e.g., clamping to [0, 1]).
- ``create_iterative_attack()``: Builds an iterative attack from a gradient
  step function, with norm-aware projection for ``linf``, ``l2``, ``l1``, ``l0``.

preconfigured.py
~~~~~~~~~~~~~~~~

Pre-instantiated attacks for immediate use without external libraries:

- ``pgd``, ``fgsm``, ``apgd``, ``fab``, ``fmn``, ``deepfool``,
  ``superdeepfool``, ``trust_region``

These are created from the ``original/`` implementations and include
``_attackbench_name`` and ``_attackbench_lib`` metadata for automatic tracking.

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
- ``targets``: ``LongTensor`` or ``None`` for targeted attacks
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
  to lists of adversarial distances
- ``best_optim_distances``: Optimal distances tracked by ``BenchModel`` during the attack
- ``adv_success``: Boolean list indicating successful adversarial examples
- ``ori_success``: Boolean list indicating originally correct predictions

**Optional metadata** (when ``include_metadata=True``):

- ``original_predictions``: Model predictions on clean inputs
- ``adversarial_predictions``: Model predictions on adversarial inputs
- ``num_forwards``: Forward pass count per sample
- ``num_backwards``: Backward pass count per sample
- ``times``: Execution time per sample
- ``hashes``: Sample identifiers

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
