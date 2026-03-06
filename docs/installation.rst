Installation
============

Requirements
------------

- Python >= 3.9, < 3.13
- PyTorch >= 2.4
- TorchVision >= 0.19
- CUDA compatible GPU (recommended)

Basic Installation
------------------

Install the base package from PyPI:

.. code-block:: bash

   pip install attackbench

.. note::

   The base installation includes core dependencies (torch, torchvision, matplotlib, pandas,
   scipy, numpy, tqdm, wandb). This is enough to run attacks using the built-in
   preconfigured attacks and analyze results.

Installation with Optional Dependencies
---------------------------------------

AttackBench provides several optional dependency groups for extended functionality:

**Attack Libraries**

Install all supported attack library wrappers (ART, Foolbox, Torchattacks, CleverHans, RobustBench):

.. code-block:: bash

   pip install "attackbench[attacks]"

.. note::

   ``adv-lib`` (Adversarial Library) is not available on PyPI. If you need adv-lib attacks,
   install it separately from its GitHub repository:
   ``pip install git+https://github.com/jeromerony/adversarial-library``

   ``deeprobust`` requires ``scipy<1.8.0`` and is incompatible with Python >= 3.10.
   Install it separately on Python 3.9 only:
   ``pip install "attackbench[deeprobust]"``

**Models**

Install model loading utilities (RobustBench, timm, transformers, pretrainedmodels):

.. code-block:: bash

   pip install "attackbench[models]"

**Metrics**

Install analysis and visualization tools (scikit-learn, seaborn, plotly, tabulate):

.. code-block:: bash

   pip install "attackbench[metrics]"

**All Dependencies**

Install everything (attacks + models + metrics, excluding deeprobust):

.. code-block:: bash

   pip install "attackbench[all]"

**Development**

Install development tools (pytest, black, isort, flake8):

.. code-block:: bash

   pip install "attackbench[dev]"

**Documentation**

Install documentation building tools (Sphinx, RTD theme):

.. code-block:: bash

   pip install "attackbench[docs]"

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/attackbench/AttackBench.git
   cd AttackBench
   pip install -e ".[dev]"

Verification
------------

Verify your installation:

.. code-block:: python

   import attackbench
   print(attackbench.__version__)
