Examples
========

This section provides practical examples of using AttackBench.

.. tip::

   A comprehensive interactive tutorial is available as a Google Colab notebook:
   `Open Tutorial in Colab <https://colab.research.google.com/drive/1rzzLRjMovcns25qOeEXt15R3L2Md_Pst?usp=sharing>`_

Running Preconfigured Attacks
-----------------------------

AttackBench includes ready-to-use attack implementations that do not require
any external attack library:

.. code-block:: python

   import torch
   import attackbench
   from attackbench.attacks import pgd, apgd, fmn, fab, deepfool

   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load model and dataset
   model = attackbench.get_model('Standard')
   model.to(device)

   dataset = attackbench.get_loader(
       dataset='cifar10',
       batch_size=128,
       num_samples=1000
   )

   # Run an attack
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device
   )

   # Analyze results (requires attackbench[metrics])
   stats = attackbench.get_stats(results, 'linf')
   print(f"ASR: {stats['asr']*100:.1f}%")

Using Library Attacks
---------------------

Use attacks from external libraries via the dynamic attack loading system
(requires ``attackbench[attacks]``):

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

   # BoMN selects the minimum-distance successful adversarial per sample
   # By definition, it achieves LocalOpt = 1.0 for all samples

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

   # Compare attacks (requires attackbench[metrics])
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
   attackbench.upload_precompiled_distances(
       attack_data=results,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd'
   )

   # Download precompiled distances from W&B
   distances = attackbench.download_precompiled_distances(
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='pgd',
       n_samples=1000
   )

   # List all available distances on W&B
   available = attackbench.list_available_distances()

W&B Caching
~~~~~~~~~~~~

By default, ``run_attack()`` checks W&B for cached results before running:

.. code-block:: python

   # Automatic caching (default: use_cached=True)
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device
   )
   # If precompiled results exist on W&B, returns them immediately

   # Disable caching to force re-running
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=apgd,
       threat_model='linf',
       device=device,
       use_cached=False
   )

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
       device=device,
       include_metadata=True
   )

   # Now includes:
   # - original_predictions, adversarial_predictions
   # - num_forwards, num_backwards (query counts)
   # - times (execution time per sample)
   # - hashes (sample identifiers)

Multi-Model Evaluation
----------------------

Evaluate an attack across multiple models:

.. code-block:: python

   models_to_test = ['Standard', 'Carmon2019Unlabeled', 'Wong2020Fast']

   for model_name in models_to_test:
       model = attackbench.get_model(model_name)
       model.to(device)

       results = attackbench.run_attack(
           model=model,
           dataset=dataset,
           attack=apgd,
           threat_model='linf',
           device=device
       )

       stats = attackbench.get_stats(results, 'linf')
       print(f"{model_name}: ASR={stats['asr']*100:.1f}%")
