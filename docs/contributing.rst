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

Install development dependencies:

.. code-block:: bash

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

1. Create a new file in ``attack_evaluation/attacks/``
2. Implement the attack following the existing patterns
3. Add tests for your implementation
4. Update the documentation

Adding New Models
-----------------

To add support for a new model:

1. Add model loading code to ``attack_evaluation/models/``
2. Update configuration files in ``exp_configs/``
3. Test the model with existing attacks

Reporting Issues
----------------

Please report issues on GitHub: https://github.com/attackbench/AttackBench/issues

License
-------

This project is licensed under the terms specified in the LICENSE file.
