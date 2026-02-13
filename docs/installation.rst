Installation
============

Requirements
------------

- Python >= 3.9, < 3.10
- PyTorch 1.12.1
- CUDA compatible GPU (recommended)

Basic Installation
------------------

Install the base package:

.. code-block:: bash

   pip install -e .

Installation with Optional Dependencies
---------------------------------------

AttackBench provides several optional dependency groups:

**Attack Libraries**

Install all supported attack library implementations:

.. code-block:: bash

   pip install -e ".[attacks]"

**Datasets**

Install dataset loading utilities:

.. code-block:: bash

   pip install -e ".[datasets]"

**Models**

Install pre-trained model utilities:

.. code-block:: bash

   pip install -e ".[models]"

**Metrics**

Install analysis and visualization tools:

.. code-block:: bash

   pip install -e ".[metrics]"

**All Dependencies**

Install everything:

.. code-block:: bash

   pip install -e ".[all]"

**Development**

Install development tools (testing, linting, formatting):

.. code-block:: bash

   pip install -e ".[dev]"

**Documentation**

Install documentation building tools:

.. code-block:: bash

   pip install -e ".[docs]"

Verification
------------

Verify your installation:

.. code-block:: python

   import attackbench
   print(attackbench.__version__)
