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

   pip install attackbenchlib

.. note::

   The base installation includes core dependencies (torch, torchvision, matplotlib, pandas,
   scipy, numpy, tqdm, wandb). This is enough to run attacks using the built-in
   preconfigured attacks and analyze results.

   The Python import name remains ``attackbench``:

   .. code-block:: python

      import attackbench

Installation with Optional Dependencies
---------------------------------------

AttackBenchLib provides several optional dependency groups for extended functionality:

**Attack Libraries**

Install all supported attack library wrappers (ART, Foolbox, Torchattacks, CleverHans):

.. code-block:: bash

   pip install "attackbenchlib[attacks]"

.. note::

   ``adv-lib`` (Adversarial Library) is not available on PyPI. If you need adv-lib attacks,
   install it separately from its GitHub repository:

   .. code-block:: bash

      pip install git+https://github.com/jeromerony/adversarial-library

   ``deeprobust`` requires ``scipy<1.8.0`` and is incompatible with Python >= 3.10.
   Install it separately on Python 3.9 only:

   .. code-block:: bash

      pip install "attackbenchlib[deeprobust]"

**Models**

Install model loading utilities (RobustBench, timm, transformers, pretrainedmodels):

.. code-block:: bash

   pip install "attackbenchlib[models]"

.. warning::

   RobustBench depends on ``autoattack``. The PyPI package ``pyautoattack`` is included
   as a dependency, but if you encounter import errors related to autoattack, install it
   manually from the official repository:

   .. code-block:: bash

      pip install git+https://github.com/fra31/auto-attack

**Metrics**

Install analysis and visualization tools (scikit-learn, seaborn, plotly, tabulate):

.. code-block:: bash

   pip install "attackbenchlib[metrics]"

**All Dependencies**

Install everything (attacks + models + metrics, excluding deeprobust):

.. code-block:: bash

   pip install "attackbenchlib[all]"

**Development**

Install development tools (pytest, black, isort, flake8):

.. code-block:: bash

   pip install "attackbenchlib[dev]"

**Documentation**

Install documentation building tools (Sphinx, RTD theme):

.. code-block:: bash

   pip install "attackbenchlib[docs]"

Google Colab
------------

On Google Colab, use the following installation commands:

.. code-block:: python

   !pip install "attackbenchlib[models,attacks]" -q
   !pip install git+https://github.com/fra31/auto-attack -q  # required for RobustBench

.. note::

   You may see red dependency conflict warnings during installation. These are caused by
   RobustBench's strict dependency pins (e.g., ``timm==1.0.9``) conflicting with Colab's
   pre-installed packages. They are harmless warnings — the library works correctly.

Development Installation
------------------------

To install from source for development:

.. code-block:: bash

   git clone https://github.com/attackbench/AttackBenchLib.git
   cd AttackBenchLib
   pip install -e ".[dev]"

Verification
------------

Verify your installation:

.. code-block:: python

   import attackbench
   print(attackbench.__version__)
