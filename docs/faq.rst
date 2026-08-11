FAQ
===

Frequently Asked Questions about AttackBenchLib.

General Questions
-----------------

What is AttackBenchLib?
~~~~~~~~~~~~~~~~~~~~~~

AttackBenchLib is a Python library that implements the **AttackBench** framework for
fairly evaluating and comparing gradient-based adversarial attacks.
It provides standardized implementations, optimality metrics, and benchmarking tools.

Why use AttackBenchLib?
~~~~~~~~~~~~~~~~~~~~~~

- **Fair Comparison**: All attacks run in the same environment with consistent settings
- **Optimality Metrics**: Beyond ASR, measure how close attacks are to optimal perturbations
- **Multiple Libraries**: Compare implementations across ART, Foolbox, Torchattacks, etc.
- **Reproducibility**: Standardized protocol and precompiled results on W&B
- **Modular Design**: Use only what you need via optional dependency groups

How is this different from existing benchmarks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBenchLib focuses on:

1. **Local Optimality**: Comparing attacks on individual samples
2. **Global Optimality**: Ranking attacks across multiple models
3. **Implementation Comparison**: Same attack algorithm from different libraries
4. **Comprehensive Metrics**: Beyond just success rate

Installation Issues
-------------------

What Python and PyTorch versions are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBenchLib supports:

- Python >= 3.9, < 3.13
- PyTorch >= 2.4
- TorchVision >= 0.19

Install the base package:

.. code-block:: bash

   pip install attackbenchlib

ImportError for optional features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBenchLib uses lazy imports. If you see an error like
``attackbench.get_stats requires the 'metrics' subpackage``, install the required extra:

.. code-block:: bash

   # For analysis features
   pip install "attackbenchlib[metrics]"

   # For library attack wrappers
   pip install "attackbenchlib[attacks]"

   # For model loading
   pip install "attackbenchlib[models]"

   # Everything at once
   pip install "attackbenchlib[all]"

How do I install adv-lib?
~~~~~~~~~~~~~~~~~~~~~~~~~~

``adv-lib`` (Adversarial Library) is not available on PyPI. Install it manually:

.. code-block:: bash

   pip install git+https://github.com/jeromerony/adversarial-library

How do I install deeprobust?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``deeprobust`` requires ``scipy<1.8.0`` and is only compatible with Python 3.9:

.. code-block:: bash

   pip install "attackbenchlib[deeprobust]"

Usage Questions
---------------

How do I create a custom attack?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a function with signature ``(model, inputs, labels, **kwargs) -> adv_inputs``:

.. code-block:: python

   def my_attack(model, inputs, labels, eps=0.3):
       # Your attack logic
       return adversarial_examples

   # Option 1: Use directly
   results = attackbench.run_attack(model, dataset, my_attack, 'linf', device, eps=0.3)

   # Option 2: Wrap with validation
   wrapped = attackbench.create_custom_attack(my_attack, attack_name="MyAttack")
   results = attackbench.run_attack(model, dataset, wrapped, 'linf', device, eps=0.3)

What threat models are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``'linf'``: :math:`\ell_\infty` norm (maximum perturbation per pixel)
- ``'l2'``: :math:`\ell_2` norm (Euclidean distance)
- ``'l1'``: :math:`\ell_1` norm (Manhattan distance)
- ``'l0'``: :math:`\ell_0` norm (number of changed pixels)

How do I save adversarial examples?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = attackbench.run_attack(
       model, dataset, attack, 'linf', device,
       save_adversarial=True
   )

   # Access adversarial examples
   adv_images = results['adv_inputs']

   # Save to disk
   torch.save(adv_images, 'adversarial_examples.pt')

Can I use my own dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, create a standard PyTorch ``DataLoader``:

.. code-block:: python

   from torch.utils.data import DataLoader

   custom_dataset = YourCustomDataset()
   loader = DataLoader(custom_dataset, batch_size=64, shuffle=False)

   results = attackbench.run_attack(model, loader, attack, 'linf', device)

How do I load models?
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Via AttackBenchLib's model registry
   model = attackbench.get_model('Standard')

   # Via RobustBench with auto-metadata
   model = attackbench.load_model('Standard', dataset='cifar10', threat_model='Linf')

MNIST checkpoints are automatically downloaded from GitHub Releases on first use
and cached in ``models/checkpoints/``.

What does ``run_attack`` return?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``run_attack`` returns minimal data:

- ``distances``: Dict per norm of ``d*``, the best perturbation found during the
  optimization (not the attack's last iterate)
- ``final_distances``: Dict per norm of the last-iterate distance (diagnostics)
- ``adv_success``: Boolean list of successful attacks
- ``ori_success``: Boolean list of samples already misclassified before the attack
- ``correct``: Boolean list of clean correctness
- ``hashes``: SHA-512 hash of each input image (always included)

Use ``include_metadata=True`` for additional data (queries, times, predictions).
Use ``save_adversarial=True`` to include adversarial tensors.

For statistics, pass results to ``attackbench.get_stats(results, threat_model)``.

Performance Questions
---------------------

Attacks are running slowly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tips for faster execution:

1. Use GPU: ``device = torch.device('cuda')``
2. Increase batch size: ``batch_size=128`` (if memory allows)
3. Enable W&B caching: ``use_cached=True`` (default) skips runs if results exist
4. Use compiled models when possible

Out of memory errors
~~~~~~~~~~~~~~~~~~~~

Solutions:

1. Reduce batch size
2. Process smaller subsets: ``num_samples=100``
3. Use CPU if GPU memory is insufficient: ``device=torch.device('cpu')``

Results and Analysis
--------------------

What is Attack Success Rate (ASR)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASR is the percentage of correctly classified samples that were successfully
misclassified by the attack:

.. math::

   \text{ASR} = \frac{\text{\# successful attacks}}{\text{\# originally correct}} \times 100\%

How is optimality computed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local optimality compares attacks per sample:

.. math::

   \text{Optimality} = \frac{\text{best distance among all attacks}}{\text{this attack's distance}}

Optimal distances are stored on W&B as **hash-based lookup tables**
(``{sha512_hash: distance}``), computed over the full dataset using all
available attacks. When evaluating a subset, each sample is matched by its
hash, ensuring correct optimality values regardless of subset size or ordering.

See :doc:`optimality` for details.

W&B Integration
---------------

Do I need a Weights & Biases account?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only if you want to:

- Upload your attack results to the shared database
- Download precompiled distances from the leaderboard
- Use automatic caching in ``run_attack()``

The core functionality (running attacks, analysis) works without W&B.

How do I authenticate with W&B?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You need to authenticate before using any W&B features:

.. code-block:: python

   # Option 1: Interactive login (recommended for local use)
   import wandb
   wandb.login()

   # Option 2: API key via environment variable (recommended for Colab/notebooks)
   # Get your API key from: https://wandb.ai/authorize
   import os
   os.environ["WANDB_API_KEY"] = "your_api_key_here"

How do I upload my results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you used ``get_loader()``, ``get_model()``, and ``get_attack()`` to set up
your pipeline, ``run_attack()`` embeds all the necessary metadata automatically:

.. code-block:: python

   results = attackbench.run_attack(model, loader, attack, threat_model='linf')

   # Metadata is extracted automatically from results['metadata']
   attackbench.upload_precompiled_distances(attack_data=results)

You can also pass metadata explicitly (e.g. for custom attacks):

.. code-block:: python

   attackbench.upload_precompiled_distances(
       attack_data=results,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='my_attack',
       attack_lib='custom'
   )

Can I use AttackBenchLib offline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. Set ``use_cached=False`` in ``run_attack()`` to skip W&B checks.
All core functionality works locally.

Is dataset sampling deterministic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. ``get_loader()`` uses a deterministic seed (``seed=0`` by default) for
subset selection. The same ``(dataset, num_samples, seed)`` combination always
returns the same samples. Pass a different ``seed`` value to obtain a different
subset.

What are sample hashes?
~~~~~~~~~~~~~~~~~~~~~~~~

``run_attack()`` always computes a SHA-512 hash of each input image's raw RGB
values. These hashes uniquely identify samples independently of their position
in the dataset or the subset used. They are used to match results against
hash-based optimal distances stored on W&B.

Contributing
------------

How can I add a new attack?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create implementation in ``attackbench/attacks/<library>/``
2. Follow the standard interface: ``(model, inputs, labels, ...) -> adv_inputs``
3. Register via a config/getter function
4. Add tests and submit a pull request

See :doc:`contributing` for details.

Where do I report bugs?
~~~~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub: https://github.com/attackbench/AttackBenchLib/issues

Citation
--------

How do I cite AttackBenchLib?
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bibtex

   @inproceedings{cina2025attackbench,
     title={Attackbench: Evaluating gradient-based attacks for adversarial examples},
     author={Cin{\`a}, Antonio Emanuele and Rony, J{\'e}r{\^o}me and Pintor, Maura and
             Demetrio, Luca and Demontis, Ambra and Biggio, Battista and
             Ayed, Ismail Ben and Roli, Fabio},
     booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
     volume={39},
     number={3},
     pages={2600--2608},
     year={2025},
     DOI={10.1609/aaai.v39i3.32263}
   }
