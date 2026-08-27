Quick Start
===========

This guide will help you get started with AttackBenchLib.

Basic Setup
-----------

First, import the required modules and set up your environment:

.. code-block:: python

   import torch
   import attackbench

   # Setup device
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

   # Load a pre-trained model (requires attackbenchlib[models])
   # Keys of the model registry are lowercase: 'standard', 'carmon_2019', ...
   model = attackbench.get_model('standard')
   model.to(device)

   # Load dataset (deterministic: same seed returns the same subset)
   dataset = attackbench.get_loader(
       dataset='cifar10',
       batch_size=128,
       num_samples=1000,
       seed=0          # default; change for a different subset
   )

You can also load models directly from RobustBench:

.. code-block:: python

   # Load a RobustBench model with auto-metadata
   model = attackbench.load_model(
       model_name='Standard',
       dataset='cifar10',
       threat_model='Linf'
   )

Running a Simple Attack
-----------------------

Using Preconfigured Attacks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBenchLib ships with pure-PyTorch preconfigured attacks that are ready to use with
the base installation:

.. code-block:: python

   from attackbench.attacks import pgd

   # Run PGD attack
   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=pgd,
       threat_model='linf',
       device=device
   )

   # Analyze results (requires attackbenchlib[metrics])
   stats = attackbench.get_stats(results, 'linf')

   print(f"Attack Success Rate: {stats['ASR']*100:.1f}%")
   print(f"Clean accuracy:      {stats['accuracy']*100:.1f}%")

The attack ran under the protocol's query budget (2000 forward+backward propagations per
sample). ``results['distances']`` holds the smallest perturbation found *during* the
optimization, which is what the optimality metric is computed on — see :doc:`optimality`.

Available preconfigured attacks:

.. code-block:: python

   from attackbench.attacks import (
       pgd,            # Projected Gradient Descent
       fgsm,           # Fast Gradient Sign Method
       apgd,           # Auto-PGD
       fab,            # Fast Adaptive Boundary
       fmn,            # Fast Minimum Norm
       deepfool,       # DeepFool
       superdeepfool,  # SuperDeepFool
       trust_region,   # Trust Region
   )

.. note::

   The preconfigured ``fmn`` attack uses Foolbox and EagerPy and therefore requires
   ``pip install "attackbenchlib[attacks]"``. The other attacks above work with the base
   installation.

Using Library Attacks
~~~~~~~~~~~~~~~~~~~~~

To use attacks from external libraries (requires ``attackbenchlib[attacks]``):

.. code-block:: python

   # List all available attacks
   all_attacks = attackbench.list_attacks()

   # List attacks for a specific threat model
   linf_attacks = attackbench.list_attacks(threat_model='linf')

   # List attacks from a specific library
   art_attacks = attackbench.list_attacks(lib='art')

   # Get a specific attack
   fmn_adv_lib = attackbench.get_attack(
       lib='adv_lib',
       attack='fmn',
       threat_model='l2'
   )

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=fmn_adv_lib,
       threat_model='l2',
       device=device
   )

Creating a Custom Attack
~~~~~~~~~~~~~~~~~~~~~~~~

You can wrap your own attack function for use with AttackBenchLib:

.. code-block:: python

   import torch.nn.functional as F

   def my_pgd(model, inputs, labels, eps=0.3, steps=40):
       adv_inputs = inputs.clone()
       alpha = eps / steps
       for _ in range(steps):
           adv_inputs.requires_grad_(True)
           loss = F.cross_entropy(model(adv_inputs), labels)
           grad = torch.autograd.grad(loss, adv_inputs)[0]
           adv_inputs = (adv_inputs + alpha * grad.sign()).detach()
           adv_inputs = torch.clamp(adv_inputs, inputs - eps, inputs + eps)
           adv_inputs = torch.clamp(adv_inputs, 0, 1)
       return adv_inputs

   # Wrap with validation and constraint enforcement
   custom_attack = attackbench.create_custom_attack(
       my_pgd,
       attack_name="MyPGD"
   )

   results = attackbench.run_attack(
       model=model,
       dataset=dataset,
       attack=custom_attack,
       threat_model='linf',
       device=device
   )

Analyzing Results
-----------------

.. code-block:: python

   # Basic statistics (requires attackbenchlib[metrics])
   stats = attackbench.get_stats(results, 'linf')

   # Compare multiple attacks
   results_pgd = attackbench.run_attack(model, dataset, pgd, 'linf', device)
   results_apgd = attackbench.run_attack(model, dataset, apgd, 'linf', device)

   comparison = attackbench.compare_attacks(
       [results_pgd, results_apgd],
       threat_model='linf'
   )

Checking a Run Against the Protocol
-----------------------------------

``scripts/paper_acceptance.py`` in the repository runs one or more attacks under the
paper's conditions and prints them with the columns of Table I, flagging protocol
violations (budget exceeded, attacks that raise, adversarials outside ``[0, 1]``, ASR
far below 100%):

.. code-block:: bash

   python scripts/paper_acceptance.py --model standard --dataset cifar10 \
       --threat-model l2 --attacks original:fmn adv_lib:fmn --reference ensemble

Next Steps
----------

- See :doc:`examples` for more complex usage scenarios
- Read :doc:`optimality` to understand the local and global optimality metrics
- Check :doc:`api/index` for detailed API documentation
- Read the full paper at https://arxiv.org/pdf/2404.19460
