Architecture
============

AttackBench is organized into several key modules that work together to provide a comprehensive benchmarking framework.

Overview
--------

The framework follows a five-stage evaluation process:

1. **Model Selection**: Select diverse robust and non-robust models
2. **Attack Execution**: Run attacks in a standardized environment
3. **Local Optimality**: Compute local optimality metrics
4. **Results Aggregation**: Aggregate results across models
5. **Global Ranking**: Rank attacks by global optimality

Core Modules
------------

attackbench
~~~~~~~~~~~

The main package providing the high-level API:

- ``run_attack()``: Execute attacks and collect raw results
- ``get_stats()``: Compute statistics from attack results
- ``bomn_attack()``: Run Best-of-MinNorm composite attacks
- ``get_model()``, ``get_loader()``, ``get_attack()``: Load components

attack_evaluation
~~~~~~~~~~~~~~~~~

Contains attack implementations and evaluation logic:

**attacks/**
  Wrappers for attacks from various libraries:
  
  - ``original/``: Custom implementations (FMN, ALMA, etc.)
  - ``art/``: IBM Adversarial Robustness Toolbox wrappers
  - ``foolbox/``: Foolbox library wrappers
  - ``torchattacks/``: Torchattacks library wrappers
  - ``adv_lib/``: Adversarial Library wrappers
  - ``cleverhans/``: CleverHans wrappers
  - ``deeprobust/``: DeepRobust wrappers

**datasets/**
  Dataset loading utilities:
  
  - CIFAR-10, CIFAR-100
  - ImageNet
  - MNIST
  - Custom dataset support

**models/**
  Model loading and management:
  
  - RobustBench integration
  - Standard models
  - Robust models
  - Custom model support

**metrics/**
  Evaluation metrics:
  
  - Distance computation (L0, L1, L2, L-infinity)
  - Attack success rate (ASR)
  - Local optimality
  - Global optimality

analysis
~~~~~~~~

Tools for analyzing and visualizing results:

- ``compile.py``: Aggregate results across experiments
- ``plot.py``: Visualization utilities
- ``plot_distances.py``: Plot robust accuracy curves
- ``utils.py``: Optimality computation helpers

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

Results Format
--------------

The ``run_attack()`` function returns a dictionary with:

**Essential data:**

- ``distances``: Dict mapping norm names to lists of distances
- ``best_optim_distances``: Optimal distances tracked during attack
- ``adv_success``: Boolean list indicating successful attacks
- ``ori_success``: Boolean list indicating originally correct predictions

**Optional metadata** (when ``include_metadata=True``):

- ``original_predictions``: Predictions on clean inputs
- ``adversarial_predictions``: Predictions on adversarial inputs
- ``num_forwards``: Forward pass count per sample
- ``num_backwards``: Backward pass count per sample
- ``times``: Execution time per sample
- ``hashes``: Sample identifiers

**Optional tensors** (when ``save_adversarial=True``):

- ``adv_inputs``: Adversarial examples
- ``inputs``: Original inputs

Extension Points
----------------

Custom Attacks
~~~~~~~~~~~~~~

Implement custom attacks by providing a callable:

.. code-block:: python

   def my_attack(model, x, y, **kwargs):
       # Your attack logic
       return adv_x

Custom Datasets
~~~~~~~~~~~~~~~

Register custom datasets in ``attack_evaluation/datasets/``:

.. code-block:: python

   from .ingredient import dataset_ingredient
   
   @dataset_ingredient.named_config
   def my_dataset():
       name = 'my_dataset'
       # Configuration...

Custom Models
~~~~~~~~~~~~~

Add model loaders in ``attack_evaluation/models/``:

.. code-block:: python

   from .ingredient import model_ingredient
   
   @model_ingredient.named_config
   def my_model():
       name = 'my_model'
       # Configuration...
