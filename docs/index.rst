Welcome to AttackBenchLib's documentation!
==========================================

**AttackBenchLib** is a Python library that implements the framework described in the
**AttackBench** paper in a modular, user-friendly way, making multiple workflows and
kinds of analysis possible through a single library.

**Leaderboard**: https://attackbench.github.io/

**Paper**: https://arxiv.org/pdf/2404.19460

**PyPI**: https://pypi.org/project/attackbenchlib/

**Tutorial Notebook**: `Open in Google Colab <https://colab.research.google.com/github/sickrichmond/AttackBench/blob/main/examples/AttackBenchLib.ipynb>`_

Overview
--------

The AttackBench framework fairly compares gradient-based attacks based on their security
evaluation curves through a five-stage process:

1. **Model Zoo**: Construct a list of diverse non-robust and robust models
2. **Attack Benchmarking**: Run every attack under the same query budget, keeping the
   best perturbation found rather than the last iterate
3. **Local Optimality**: Compare attacks using the novel local optimality metric
4. **Global Optimality**: Aggregate optimality results from all models
5. **Ranking**: Rank attacks based on their average (global) optimality

Features
--------

- Standardized evaluation protocol for adversarial attacks, with the paper's query
  budget (2000 forward+backward propagations per sample) enforced by default
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

   pip install attackbenchlib

   # With all optional dependencies
   pip install "attackbenchlib[all]"

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   external_models
   architecture
   optimality
   api/index
   examples
   faq
   contributing
   releasing

Citation
--------

If you use AttackBenchLib in your research, please cite:

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

AttackBenchLib has been partially developed with the support of:

- European Union's **ELSA -- European Lighthouse on Secure and Safe AI**, Horizon Europe, grant agreement No. 101070617
- **Sec4AI4Sec - Cybersecurity for AI-Augmented Systems**, Horizon Europe, grant agreement No. 101120393

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
