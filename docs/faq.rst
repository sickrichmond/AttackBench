FAQ
===

Frequently Asked Questions about AttackBench.

General Questions
-----------------

What is AttackBench?
~~~~~~~~~~~~~~~~~~~~

AttackBench is a framework for fairly evaluating and comparing gradient-based adversarial attacks. It provides standardized implementations, optimality metrics, and benchmarking tools.

Why use AttackBench?
~~~~~~~~~~~~~~~~~~~~

- **Fair Comparison**: All attacks run in the same environment with consistent settings
- **Optimality Metrics**: Beyond ASR, measure how close attacks are to optimal
- **Multiple Libraries**: Compare implementations across ART, Foolbox, Torchattacks, etc.
- **Reproducibility**: Standardized protocol and precompiled results database

How is this different from existing benchmarks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBench focuses on:

1. **Local Optimality**: Comparing attacks on individual samples
2. **Global Optimality**: Ranking attacks across multiple models
3. **Implementation Comparison**: Same attack from different libraries
4. **Comprehensive Metrics**: Beyond just success rate

Installation Issues
-------------------

ImportError: No module named 'torch'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AttackBench requires PyTorch 1.12.1. Install it first:

.. code-block:: bash

   pip install torch==1.12.1 torchvision==0.13.1

Then install AttackBench:

.. code-block:: bash

   pip install -e .

Conflicts with numpy versions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some attack libraries require specific numpy versions. Use the version constraints:

.. code-block:: bash

   pip install "numpy>=1.21.0,<1.25.0"

Can't install all attack libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You don't need all libraries. Install only what you need:

.. code-block:: bash

   # Just the base package
   pip install -e .
   
   # Add specific attack libraries
   pip install foolbox torchattacks

Usage Questions
---------------

How do I create a custom attack?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a function with signature ``(model, x, y, **kwargs) -> adv_x``:

.. code-block:: python

   def my_attack(model, x, y, eps=0.3):
       # Your attack logic
       return adversarial_examples

   results = run_attack(model, dataset, my_attack, 'linf', device, eps=0.3)

What threat models are supported?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``'linf'``: L-infinity norm (maximum perturbation per pixel)
- ``'l2'``: L2 norm (Euclidean distance)
- ``'l1'``: L1 norm (Manhattan distance)
- ``'l0'``: L0 norm (number of changed pixels)

How do I save adversarial examples?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   results = run_attack(
       model, dataset, attack, 'linf', device,
       save_adversarial=True
   )
   
   # Access adversarial examples
   adv_images = results['adv_inputs']
   
   # Save to disk
   torch.save(adv_images, 'adversarial_examples.pt')

Can I use my own dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, create a PyTorch DataLoader:

.. code-block:: python

   from torch.utils.data import DataLoader
   
   custom_dataset = YourCustomDataset()
   loader = DataLoader(custom_dataset, batch_size=64, shuffle=False)
   
   results = run_attack(model, loader, attack, 'linf', device)

Performance Questions
---------------------

Attacks are running slowly
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tips for faster execution:

1. Use GPU: ``device = torch.device('cuda')``
2. Increase batch size: ``batch_size=128`` (if memory allows)
3. Reduce iterations for iterative attacks
4. Use compiled models when possible

Out of memory errors
~~~~~~~~~~~~~~~~~~~~

Solutions:

1. Reduce batch size
2. Process smaller subsets: ``num_samples=100``
3. Enable gradient checkpointing if available
4. Use CPU if GPU memory is insufficient

How do I parallelize across multiple GPUs?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, AttackBench runs on a single device. For multi-GPU:

.. code-block:: python

   # Split dataset manually
   results = []
   for gpu_id in range(num_gpus):
       device = torch.device(f'cuda:{gpu_id}')
       dataset_split = get_split(dataset, gpu_id, num_gpus)
       results.append(run_attack(model, dataset_split, attack, 'linf', device))

Results and Analysis
--------------------

What is Attack Success Rate (ASR)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ASR is the percentage of correctly classified samples that were successfully misclassified by the attack:

.. math::

   ASR = \\frac{\\text{# successful attacks}}{\\text{# originally correct}} \\times 100\\%

How is optimality computed?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local optimality compares attacks per sample:

.. math::

   \\text{Optimality} = \\frac{\\text{best distance among all attacks}}{\\text{this attack's distance}}

See :doc:`optimality` for details.

Can I access raw perturbations?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, when ``save_adversarial=True``:

.. code-block:: python

   results = run_attack(model, dataset, attack, 'linf', device, 
                        save_adversarial=True)
   
   perturbations = results['adv_inputs'] - results['inputs']

W&B Integration
---------------

Do I need a Weights & Biases account?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only if you want to:

- Upload your attack results to the shared database
- Download precompiled distances from the leaderboard

The core functionality works without W&B.

How do I upload my results?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from attackbench import upload_precompiled_distances
   
   upload_precompiled_distances(
       attack_data=results,
       dataset='cifar10',
       threat_model='linf',
       model_name='Standard',
       attack_name='my_attack'
   )

Can I use AttackBench offline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes. W&B features are optional. Run attacks and analyze results locally without internet.

Contributing
------------

How can I add a new attack?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create implementation in ``attack_evaluation/attacks/``
2. Follow the standard interface
3. Add tests
4. Submit a pull request

See :doc:`contributing` for details.

How can I add a new model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add model loading code in ``attack_evaluation/models/``
2. Register it in the configuration
3. Test with existing attacks
4. Submit a pull request

Where do I report bugs?
~~~~~~~~~~~~~~~~~~~~~~~~

Open an issue on GitHub: https://github.com/attackbench/AttackBench/issues

Citation
--------

How do I cite AttackBench?
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
