Examples
========

This section provides practical examples of using AttackBench.

Custom Attack Implementation
-----------------------------

Creating Your Own Attack
~~~~~~~~~~~~~~~~~~~~~~~~

Custom attacks must be callables that accept ``(model, x, y, **kwargs)`` and return adversarial examples:

.. code-block:: python

   import torch

   def my_custom_attack(model, x, y, eps=0.3, alpha=0.01, steps=40):
       """
       Simple PGD-like attack implementation.
       
       Args:
           model: Target model
           x: Input images (batch)
           y: True labels
           eps: Maximum perturbation
           alpha: Step size
           steps: Number of iterations
       
       Returns:
           Adversarial examples
       """
       adv_x = x.clone().detach()
       
       for _ in range(steps):
           adv_x.requires_grad = True
           output = model(adv_x)
           loss = torch.nn.CrossEntropyLoss()(output, y)
           loss.backward()
           
           # PGD update
           adv_x = adv_x + alpha * adv_x.grad.sign()
           adv_x = torch.clamp(adv_x, x - eps, x + eps)
           adv_x = torch.clamp(adv_x, 0, 1).detach()
       
       return adv_x

   # Use your custom attack
   from attackbench import run_attack, get_stats

   results = run_attack(
       model=model,
       dataset=dataset,
       attack=my_custom_attack,
       threat_model='linf',
       device=device,
       eps=0.3,  # Custom parameter
       alpha=0.01,
       steps=40
   )

   stats = get_stats(results, 'linf')
   print(f"Custom Attack ASR: {stats['ASR']*100:.1f}%")

BoMN: Best-of-MinNorm Attack
-----------------------------

Run multiple attacks and select the best result per sample:

.. code-block:: python

   from attackbench import bomn_attack
   from attackbench.attacks import pgd, apgd, deepfool
   import numpy as np

   # Run BoMN with 3 attacks
   results_bomn = bomn_attack(
       model=model,
       dataset=dataset,
       attacks=[pgd, apgd, deepfool],
       threat_model='linf',
       device=device,
       verbose=True
   )

   # Analyze which attack won for each sample
   attack_names = results_bomn['attack_names']
   best_indices = np.array(results_bomn['best_attack_indices'])
   n_successful = sum(results_bomn['adv_success'])

   print("\nWins per attack:")
   for i, name in enumerate(attack_names):
       wins = (best_indices == i).sum()
       pct = 100.0 * wins / n_successful if n_successful > 0 else 0.0
       print(f"  {name}: {wins} samples ({pct:.1f}%)")

W&B Integration
---------------

Upload and Download Attack Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from attackbench import upload_precompiled_distances, download_precompiled_distances

   # Upload your attack results to W&B
   upload_precompiled_distances(
       attack_data=results,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd',
       overwrite=True
   )

   # Download precompiled distances from W&B
   distances = download_precompiled_distances(
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd',
       n_samples=100
   )

   print(f"Downloaded: ASR={distances['ASR']:.2%}")

Running Benchmarks via CLI
---------------------------

CIFAR-10 Benchmarks
~~~~~~~~~~~~~~~~~~~

L-infinity attack on CIFAR-10:

.. code-block:: bash

   python -m attack_evaluation.run -F results_dir/ with \
       model.Standard \
       attack.pgd \
       attack.threat_model="linf" \
       dataset.cifar10 \
       dataset.num_samples=1000 \
       dataset.batch_size=128

L2 attack on CIFAR-10:

.. code-block:: bash

   python -m attack_evaluation.run -F results_dir/ with \
       model.augustin_2020 \
       attack.adv_lib_fmn \
       attack.threat_model="l2" \
       dataset.num_samples=1000 \
       dataset.batch_size=64

ImageNet Benchmarks
~~~~~~~~~~~~~~~~~~~

L-infinity attack on ImageNet:

.. code-block:: bash

   python -m attack_evaluation.run -F results_dir/ with \
       model.resnet50 \
       attack.apgd \
       attack.threat_model="linf" \
       dataset.imagenet \
       dataset.num_samples=1000

Detailed Statistics
-------------------

Comprehensive Attack Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from attackbench import get_stats

   # Get comprehensive statistics
   stats = get_stats(results, 'linf', include_optimality=False)

   print("Attack Statistics:")
   print(f"  Attack Success Rate (ASR): {stats['ASR']*100:.1f}%")
   print(f"  Model Accuracy: {stats['accuracy']*100:.1f}%")
   print(f"  Mean distance: {stats['linf_mean_distance']:.6f}")
   print(f"  Median distance: {stats['linf_median_distance']:.6f}")
   print(f"  Max distance: {stats['linf_max_distance']:.6f}")
   print(f"  Min distance: {stats['linf_min_distance']:.6f}")
   
   # Distance statistics for successful attacks only
   print(f"\nSuccessful attacks only:")
   print(f"  Mean distance: {stats['linf_mean_successful_distance']:.6f}")
   print(f"  Median distance: {stats['linf_median_successful_distance']:.6f}")

Multi-Model Evaluation
----------------------

Evaluate an attack across multiple models:

.. code-block:: python

   from robustbench import load_model
   
   models = [
       ('Standard', 'Standard'),
       ('Carmon2019', 'Carmon2019Unlabeled'),
       ('Wong2020', 'Wong2020Fast'),
   ]

   print(f"{'Model':<20} {'ASR':<8} {'Mean Dist':<12}")
   print("-" * 40)

   for name, model_id in models:
       model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf')
       model.to(device)
       
       results = run_attack(
           model=model,
           dataset=dataset,
           attack=pgd,
           threat_model='linf',
           device=device
       )
       
       stats = get_stats(results, 'linf')
       print(f"{name:<20} {stats['ASR']*100:<8.1f} {stats['linf_mean_distance']:<12.6f}")
