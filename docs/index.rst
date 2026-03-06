Welcome to AttackBench's documentation!
========================================

**AttackBench** is a Python framework for evaluating gradient-based adversarial attacks.
It provides a systematic and reproducible protocol for comparing different attack implementations
based on their security evaluation curves and local optimality metrics.

**Leaderboard**: https://attackbench.github.io/

**Paper**: https://arxiv.org/pdf/2404.19460

**PyPI**: https://pypi.org/project/attackbench/

Overview
--------

The AttackBench framework fairly compares gradient-based attacks based on their security
evaluation curves through a five-stage process:

1. **Model Selection**: Construct a list of diverse non-robust and robust models
2. **Attack Execution**: Define a systematic environment for testing attacks
3. **Local Optimality**: Compare attacks using the novel local optimality metric
4. **Results Aggregation**: Aggregate optimality results from all models
5. **Global Ranking**: Rank attacks based on their average (global) optimality

Features
--------

- Standardized evaluation protocol for adversarial attacks
- Preconfigured attacks (PGD, FGSM, APGD, FAB, FMN, DeepFool, SuperDeepFool, Trust Region)
- Support for multiple attack libraries (ART, Foolbox, Torchattacks, CleverHans, adv-lib, DeepRobust)
- Local and global optimality metrics
- Best-of-MinNorm (BoMN) composite attack
- Custom attack integration with ``create_custom_attack()``
- W&B integration for sharing and caching precompiled results
- Modular design with optional dependency groups

Quick Install
-------------

.. code-block:: bash

   pip install attackbench

   # With all optional dependencies
   pip install "attackbench[all]"

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   architecture
   optimality
   api/index
   examples
   faq
   contributing

Citation
--------

If you use AttackBench in your research, please cite:

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

Acknowledgements
----------------

AttackBench has been partially developed with the support of:

- European Union's **ELSA -- European Lighthouse on Secure and Safe AI**, Horizon Europe, grant agreement No. 101070617
- **Sec4AI4Sec - Cybersecurity for AI-Augmented Systems**, Horizon Europe, grant agreement No. 101120393

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
