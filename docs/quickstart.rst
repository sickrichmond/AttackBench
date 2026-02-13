Quick Start
===========

This guide will help you get started with AttackBench.

Basic Setup
-----------

First, import the required modules and setup your environment:

.. code-block:: python

   import torch
   from attackbench import run_attack, get_stats
   from robustbench import load_model
   from attackbench import get_loader

   # Setup device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load a pre-trained model
   model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
   model.to(device)

   # Load dataset
   dataset = get_loader(
       dataset='cifar10',
       batch_size=100,
       num_samples=100,
       random_subset=True
   )

Running a Simple Attack
~~~~~~~~~~~~~~~~~~~~~~~

Run a PGD attack and analyze the results:

.. code-block:: python

   from attackbench.attacks import pgd

   # Run PGD attack
   results = run_attack(
       model=model,
       dataset=dataset,
       attack=pgd,
       threat_model='linf',
       device=device
   )

   # Get attack statistics
   stats = get_stats(results, 'linf')

   print(f"Attack Success Rate: {stats['ASR']*100:.1f}%")
   print(f"Mean L-infinity distance: {stats['linf_mean_distance']:.4f}")
   print(f"Median L-infinity distance: {stats['linf_median_distance']:.4f}")

Using the Command Line
----------------------

AttackBench provides a command-line interface for running benchmarks using Sacred:

.. code-block:: bash

   # Run FMN attack from adv_lib against augustin_2020 model
   python -m attack_evaluation.run -F results_dir/ with \
       model.augustin_2020 \
       attack.adv_lib_fmn \
       attack.threat_model="l2" \
       dataset.num_samples=1000 \
       dataset.batch_size=64 \
       seed=42

Command breakdown:

- ``-F results_dir/``: Directory where results will be saved
- ``with``: Sacred keyword for configuration
- ``model.augustin_2020``: Target model to attack
- ``attack.adv_lib_fmn``: FMN attack from adv_lib library
- ``attack.threat_model="l2"``: L2 threat model
- ``dataset.num_samples=1000``: Number of samples to attack
- ``dataset.batch_size=64``: Batch size for processing
- ``seed=42``: Random seed for reproducibility

Available Attacks
-----------------

AttackBench supports multiple attack implementations from various libraries:

.. code-block:: python

   from attackbench.attacks import (
       pgd,        # Projected Gradient Descent
       fgsm,       # Fast Gradient Sign Method
       apgd,       # Auto-PGD
       deepfool,   # DeepFool
       cw_l2,      # Carlini-Wagner L2
       fab,        # Fast Adaptive Boundary
   )

Comparing Multiple Attacks
--------------------------

.. code-block:: python

   attacks_to_compare = [
       ('PGD', pgd),
       ('FGSM', fgsm),
       ('APGD', apgd)
   ]

   print(f"{'Attack':<10} {'ASR':<8} {'Mean Linf':<12}")
   print("-" * 30)

   for name, attack in attacks_to_compare:
       results = run_attack(
           model=model,
           dataset=dataset,
           attack=attack,
           threat_model='linf',
           device=device
       )
       stats = get_stats(results, 'linf')
       print(f"{name:<10} {stats['ASR']*100:<8.1f} {stats['linf_mean_distance']:<12.6f}")

Next Steps
----------

- See :doc:`api/index` for detailed API documentation
- Check :doc:`examples` for more complex usage scenarios including custom attacks
- Read the full paper at https://arxiv.org/pdf/2404.19460
