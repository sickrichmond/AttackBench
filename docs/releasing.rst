Releasing
=========

The release workflow builds one immutable wheel/sdist pair, tests it, and publishes
that same artifact with PyPI's trusted-publishing flow. Do not upload a separately
built local artifact.

One-time repository setup
-------------------------

Create protected GitHub environments named testpypi and pypi. Configure a GitHub
Actions trusted publisher for the attackbenchlib project on both indexes with:

* owner: sickrichmond
* repository: AttackBench
* workflow: release.yml
* environment: testpypi or pypi, matching the target index

Require reviewer approval for the pypi environment. The publishing jobs alone receive
the OIDC id-token: write permission; build and test jobs remain read-only.

Pre-release checks
------------------

From a clean checkout:

.. code-block:: bash

   python -m venv .venv
   . .venv/bin/activate
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   pip install -e ".[attacks,models,dev,docs]"
   pytest
   sphinx-build -W --keep-going -b html docs docs/_build/html
   python -m build
   twine check dist/*

Run attackbench-acceptance on representative real checkpoints. At minimum, use
one fixed-budget and one minimum-norm attack for each supported dataset/norm release
claim. Use a CUDA runner for the full CIFAR-10 and ImageNet matrix. For paper-comparable
optimality, use --reference wandb only after the W&B d* envelopes have been repopulated
and hash-matched.

Test the ``torchattacks`` extra in a separate environment because upstream pins
``requests~=2.25.1``; it is intentionally not part of ``attacks`` or ``all``.

TestPyPI
--------

Run the publish workflow manually with repository=testpypi. The workflow:

1. runs the extras-enabled tests;
2. builds and checks both distributions;
3. publishes through the protected testpypi environment; and
4. installs the exact uploaded version from TestPyPI in a clean job.

Inspect that run before creating the final tag. Distribution filenames are immutable:
if the target version already exists on TestPyPI, increment to a pre-release version
rather than enabling duplicate-skipping.

PyPI
----

After TestPyPI and the acceptance matrix pass, create the tag that exactly matches
the project version:

.. code-block:: bash

   git tag -s v2.0.2 -m "AttackBenchLib 2.0.2"
   git push origin v2.0.2

The tag starts the same build/test workflow and sends its artifact to the protected
pypi environment. A maintainer may instead manually select repository=pypi from main;
the environment approval still applies. PyPI releases cannot be overwritten, so stop
before environment approval if the artifact or metadata is wrong.
