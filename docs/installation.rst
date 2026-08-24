Installation
============

Requirements
------------

- Python >= 3.9
- PyTorch >= 2.4
- TorchVision >= 0.19
- CUDA compatible GPU (recommended)

Basic Installation
------------------

Install the base package from PyPI:

.. code-block:: bash

   pip install attackbenchlib

.. note::

   The base installation includes core dependencies (torch, torchvision, numpy, tqdm,
   wandb). This is enough to run the built-in preconfigured attacks and analyze the
   results.

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

   ``adv-lib`` (Adversarial Library) is not on PyPI, and it depends on ``visdom``, whose
   ``setup.py`` imports ``pkg_resources`` — removed from setuptools 81+. Build ``visdom``
   against an older setuptools first:

   .. code-block:: bash

      pip install "setuptools<81" wheel
      pip install --no-build-isolation visdom
      pip install "attackbenchlib[adv_lib]"

   The paper ranks AdvLib's implementations among the best, so an installation without it
   benchmarks weaker re-implementations of the same attacks.

   ``deeprobust`` requires ``scipy<1.8.0`` and is incompatible with Python >= 3.10.
   Install it separately on Python 3.9 only:

   .. code-block:: bash

      pip install "attackbenchlib[deeprobust]"

**Models**

Install model loading utilities (RobustBench, Pillow; RobustBench pulls in
whatever a given model needs, such as timm):

.. code-block:: bash

   pip install "attackbenchlib[models]"

.. warning::

   RobustBench depends on ``autoattack``. The PyPI package ``pyautoattack`` is included
   as a dependency, but if you encounter import errors related to autoattack, install it
   manually from the official repository:

   .. code-block:: bash

      pip install git+https://github.com/fra31/auto-attack

**Metrics**

Enable the analysis subpackage (its code only needs numpy, already a core dependency,
so the extra is empty and exists to keep the install command valid):

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

.. code-block:: text

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
