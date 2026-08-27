Contributing
============

We welcome contributions to AttackBenchLib!

How to Contribute
-----------------

1. Fork the repository
2. Create a feature branch (``git checkout -b feature/amazing-feature``)
3. Make your changes
4. Run tests to ensure nothing breaks
5. Commit your changes (``git commit -m 'Add amazing feature'``)
6. Push to the branch (``git push origin feature/amazing-feature``)
7. Open a Pull Request

Development Setup
-----------------

Clone and install in development mode:

.. code-block:: bash

   git clone https://github.com/attackbench/AttackBench.git
   cd AttackBench
   pip install -e ".[dev]"

Code Style
----------

We follow PEP 8 guidelines. Use the provided tools:

.. code-block:: bash

   # Format code
   black .
   
   # Sort imports
   isort .
   
   # Lint
   flake8 .

Testing
-------

Run the test suite (CPU only, no network, seconds):

.. code-block:: bash

   pytest

To check a change against the benchmark protocol itself — query budget, best-iterate
distances, failure indicators — use the acceptance script on a real model:

.. code-block:: bash

   python scripts/paper_acceptance.py --model standard --dataset cifar10 \
       --threat-model l2 --attacks original:fmn adv_lib:fmn --reference ensemble

Adding New Attacks
------------------

To add a new attack implementation:

1. Create a new file in ``attackbench/attacks/<library>/`` (or ``attackbench/attacks/original/``
   for native implementations)
2. Implement the attack following the standard interface:
   ``(model, inputs, labels, targets=None, targeted=False, **kwargs) -> adv_inputs``
3. Register the attack with a config function returning a ``dict`` of its parameters and
   a matching getter function — see ``attackbench/attacks/README.md`` for the template
4. Add tests for your implementation
5. Update the documentation

.. note::

   ``list_attacks()`` only advertises attacks it can actually build, so check that your
   new attack shows up for every threat model it supports:

   .. code-block:: python

      attackbench.list_attacks(threat_model='l2', lib='<your library>')

Adding New Models
-----------------

To add support for a new model:

1. Add model loading code to ``attackbench/models/``
2. Register it in ``MODEL_CONFIGS`` in ``attackbench/models/registry.py``
3. Test the model with existing attacks

.. note::

   MNIST checkpoints are automatically downloaded from GitHub Releases on first use
   and cached locally in ``models/checkpoints/``. No manual download is required.

Building Documentation
----------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs/
   make html

The built documentation will be available in ``docs/_build/html/``.

Reporting Issues
----------------

Please report issues on GitHub: https://github.com/attackbench/AttackBench/issues

Contact
-------

Feel free to contact us by creating an issue, a pull request, or by email
at ``antonio.cina@unige.it``.

License
-------

AttackBench is distributed under the MIT License;
see the repository's ``LICENSE`` file. Bundled and adapted components are
documented in ``THIRD_PARTY_NOTICES.md`` and remain subject to their applicable
upstream terms.
