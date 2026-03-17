Contributing
============

We welcome contributions to AttackBench!

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

Run the test suite:

.. code-block:: bash

   pytest

Adding New Attacks
------------------

To add a new attack implementation:

1. Create a new file in ``attackbench/attacks/<library>/`` (or ``attackbench/attacks/original/``
   for native implementations)
2. Implement the attack following the standard interface:
   ``(model, inputs, labels, targets=None, targeted=False, **kwargs) -> adv_inputs``
3. Register the attack via a config/getter function in the library's submodule
4. Add tests for your implementation
5. Update the documentation

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

This project is licensed under the terms specified in the LICENSE file.
