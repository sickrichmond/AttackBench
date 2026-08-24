Examples
========

This section provides practical examples of using AttackBenchLib.

.. tip::

   A comprehensive interactive tutorial is available as a Google Colab notebook:
   `Open Tutorial in Colab <https://colab.research.google.com/github/sickrichmond/AttackBench/blob/main/examples/AttackBenchLib.ipynb>`_

Running Preconfigured Attacks
-----------------------------

AttackBenchLib includes ready-to-use attack implementations that do not require
any external attack library:

.. code-block:: python

   import torch
   import attackbench
   from attackbench.attacks import pgd, apgd, fmn, fab, deepfool

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load model and dataset
   model = attackbench.get_model('standard')
   model.to(device)

   dataset = attackbench.get_loader(
       dataset='cifar10',
       batch_size=128,
       num_samples=1000,
       seed=0          # deterministic subset selection
   )

   # Run an attack
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device
   )

   # Analyze results (requires attackbenchlib[metrics])
   stats = attackbench.get_stats(results, 'linf')
   print(f"ASR: {stats['ASR']*100:.1f}%")

Using Library Attacks
---------------------

Use attacks from external libraries via the dynamic attack loading system
(requires ``attackbenchlib[attacks]``):

.. code-block:: python

   # Discover available attacks
   all_attacks = attackbench.list_attacks()
   print(f"Total available attacks: {len(all_attacks)}")

   # Filter by threat model or library
   linf_attacks = attackbench.list_attacks(threat_model='linf')
   foolbox_attacks = attackbench.list_attacks(lib='foolbox')

   # Instantiate and run an attack from a specific library
   art_pgd = attackbench.get_attack(lib='art', attack='pgd', threat_model='linf')

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=art_pgd,
       threat_model='linf',
       device=device
   )

Custom Attack Implementation
-----------------------------

Creating Your Own Attack
~~~~~~~~~~~~~~~~~~~~~~~~

Custom attacks must be callables that accept ``(model, inputs, labels, **kwargs)``
and return adversarial examples:

.. code-block:: python

   import torch
   import torch.nn.functional as F

   def my_custom_pgd(model, inputs, labels, eps=0.3, alpha=0.01, steps=40):
       """
       Simple PGD attack implementation.

       Args:
           model: Target model
           inputs: Input images (batch)
           labels: True labels
           eps: Maximum perturbation
           alpha: Step size
           steps: Number of iterations

       Returns:
           Adversarial examples
       """
       adv_inputs = inputs.clone().detach()

       for _ in range(steps):
           adv_inputs.requires_grad_(True)
           loss = F.cross_entropy(model(adv_inputs), labels)
           grad = torch.autograd.grad(loss, adv_inputs)[0]

           adv_inputs = (adv_inputs + alpha * grad.sign()).detach()
           adv_inputs = torch.clamp(adv_inputs, inputs - eps, inputs + eps)
           adv_inputs = torch.clamp(adv_inputs, 0, 1)

       return adv_inputs

Using ``create_custom_attack`` for validation:

.. code-block:: python

   # Wrap with input validation and constraint enforcement
   custom_attack = attackbench.create_custom_attack(
       my_custom_pgd,
       validate_inputs=True,
       enforce_constraints=True,
       attack_name="MyPGD"
   )

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=custom_attack,
       threat_model='linf',
       device=device,
       eps=0.3,
       alpha=0.01,
       steps=40
   )

Or use the attack function directly — ``run_attack`` will automatically
filter kwargs to match the function's signature:

.. code-block:: python

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=my_custom_pgd,
       threat_model='linf',
       device=device,
       eps=0.3
   )

BoMN: Best-of-MinNorm Attack
-----------------------------

Run multiple attacks and select the best result per sample:

.. code-block:: python

   from attackbench.attacks import pgd, apgd, fmn

   results_bomn = attackbench.bomn_attack(
       model=model,
       dataset=dataset,
       attacks=[pgd, apgd, fmn],
       threat_model='linf',
       device=device
   )

   # BoMN keeps, for each sample, the smallest perturbation any of its components found.
   # It is the lower envelope of *those* attacks, so it scores 1.0 against them; against
   # the benchmark-wide envelope on W&B it scores below 1.0 like any other attack.
   # Every component runs through run_attack, so the query budget applies to each of them.

   # Analyze which attack won for each sample
   import numpy as np

   attack_names = results_bomn['attack_names']
   best_indices = np.array(results_bomn['best_attack_indices'])
   n_successful = sum(results_bomn['adv_success'])

   print("Wins per attack:")
   for i, name in enumerate(attack_names):
       wins = (best_indices == i).sum()
       pct = 100.0 * wins / n_successful if n_successful > 0 else 0.0
       print(f"  {name}: {wins} samples ({pct:.1f}%)")

Comparing Multiple Attacks
--------------------------

.. code-block:: python

   from attackbench.attacks import pgd, apgd, fab

   attacks_to_compare = {
       'PGD': pgd,
       'APGD': apgd,
       'FAB': fab
   }

   all_results = {}
   for name, attack in attacks_to_compare.items():
       all_results[name] = attackbench.run_attack(
           model=model,
           dataset=dataset,
           attack=attack,
           threat_model='linf',
           device=device
       )

   # Compare attacks (requires attackbenchlib[metrics])
   comparison = attackbench.compare_attacks(
       list(all_results.values()),
       threat_model='linf'
   )

W&B Integration
---------------

Authentication
~~~~~~~~~~~~~~

AttackBench uses Weights & Biases to store and share precompiled attack distances.
To use W&B features, you need to authenticate first:

.. code-block:: python

   # Option 1: Interactive login (recommended for local use)
   # Credentials are saved in ~/.netrc for future sessions
   import wandb
   wandb.login()

   # Option 2: API key via environment variable (recommended for Colab/notebooks)
   # Get your API key from: https://wandb.ai/authorize
   import os
   os.environ["WANDB_API_KEY"] = "your_api_key_here"

.. note::

   W&B authentication is only required for database features (uploading, downloading,
   caching results). All core functionality works without it.

Upload and Download Attack Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Upload your attack results to W&B
   # Metadata is extracted automatically from results['metadata']
   attackbench.upload_precompiled_distances(attack_data=results)

   # Or pass metadata explicitly (e.g. for custom attacks)
   attackbench.upload_precompiled_distances(
       attack_data=results,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd',
       attack_lib='foolbox'
   )

   # Download precompiled distances from W&B (attack_lib identifies the implementation)
   distances = attackbench.download_precompiled_distances(
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd',
       attack_lib='foolbox',
       n_samples=1000
   )

Optimal Distances
~~~~~~~~~~~~~~~~~

Hash-based optimal distances (best-known perturbations for every image in the
full dataset) can be uploaded and downloaded via W&B. They are stored as
``{sha512_hash: distance}`` dictionaries, so any subset of the dataset can be
matched regardless of ordering or size.

.. code-block:: python

   # Download optimal distances for a given configuration
   optimal = attackbench.download_optimal_distances(
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard'
   )

   # Upload an envelope you computed yourself
   attackbench.upload_optimal_distances(
       optimal_data=optimal,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard'
   )

   # Or fold one run into the published envelope: the per-hash minimum is taken
   # against what is already there and the artifact is refreshed. This is stage 5 of
   # the framework — adding an attack does not require re-running the previous ones.
   attackbench.update_optimal_distances(results)

W&B Caching
~~~~~~~~~~~~

``run_attack()`` always executes the attack you pass it. Opt in to ``use_cached=True``
to reuse results already published on W&B:

.. code-block:: python

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device,
       use_cached=True
   )

A cached artifact is only reused when its per-sample SHA-512 hashes match your samples
exactly; on any mismatch (different subset, seed or preprocessing) a warning is emitted
and the attack runs normally.

Saving Results to Disk
----------------------

.. code-block:: python

   # Save results as JSON + .pt files
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device,
       save_results=True,
       output_dir='./my_results/'
   )

   # Include adversarial examples in output
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device,
       save_adversarial=True
   )

   # Access adversarial examples
   adv_images = results['adv_inputs']

Including Metadata
------------------

By default, ``run_attack()`` returns minimal data. Request additional metadata:

.. code-block:: python

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device
   )

   # Always included, no flag needed:
   # - original_predictions, adversarial_predictions
   # - num_forwards, num_backwards (query counts, capped by query_budget)
   # - times (execution time per batch)
   # - box_failures, batch_failures (attack failure indicators)
   # - hashes (SHA-512 per image, for hash-based optimality matching)

Multi-Model Evaluation
----------------------

Evaluate an attack across multiple models:

.. code-block:: python

   models_to_test = ['standard', 'carmon_2019', 'wong_2020']

   for model_name in models_to_test:
       model = attackbench.get_model(model_name)  # registry keys are lowercase
       model.to(device)

       results = attackbench.run_attack(
           model=model,
           dataset=dataset,
           attack=apgd,
           threat_model='linf',
           device=device
       )

       stats = attackbench.get_stats(results, 'linf')
       print(f"{model_name}: ASR={stats['ASR']*100:.1f}%")
