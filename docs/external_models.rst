External-checkpoint models
==========================

AttackBench is MIT-licensed, but a model's code and pretrained parameters are
separate copyrighted works. The stutz_2020 and xiao_2020 registry entries
therefore use independent MIT architecture implementations while keeping their
upstream checkpoints outside the package. AttackBench loads these files with
PyTorch's restricted weights_only=True mode and verifies pinned SHA-256 values.

Stutz 2020 CCAT
---------------

The author's terms permit the upstream software for noncommercial scientific
research, noncommercial education, and noncommercial artistic projects only.
Commercial use is prohibited, the notice must be retained, and research using
it must cite the paper. Read the complete terms in AttackBench's third-party
notices_ and at the upstream Stutz repository (stutz_repo_).

After confirming that your use complies, AttackBench can fetch the
author-hosted checkpoint into PyTorch's cache:

.. code-block:: python

   import attackbench

   model = attackbench.get_model("stutz_2020", accept_license=True)

For command-line experiments, use the equivalent acknowledgement:

.. code-block:: bash

   ATTACKBENCH_ACCEPT_STUTZ2020_LICENSE=1 attackbench-acceptance \
       --model stutz_2020 --dataset cifar10 --threat-model linf ...

An already-downloaded, extracted classifier.pth.tar can instead be supplied
through checkpoint_path or ATTACKBENCH_STUTZ2020_CHECKPOINT. The
acknowledgement is still required.

Xiao 2020 k-WTA
---------------

The upstream robustness_workshop repository and its released checkpoint do not
state a license. AttackBench therefore does not copy, redistribute, or
automatically download that file. Obtain any permission your use requires,
download the kwta_spresnet18_0.1_cifar_adv.pth checkpoint yourself from the
upstream Xiao release (xiao_release_), and pass its path explicitly:

.. code-block:: python

   import attackbench

   model = attackbench.get_model(
       "xiao_2020",
       checkpoint_path="/path/to/kwta_spresnet18_0.1_cifar_adv.pth",
   )

For command-line experiments, set ATTACKBENCH_XIAO2020_CHECKPOINT to that path.

The independent architecture code is MIT-licensed with the rest of
AttackBench. This does not grant any rights in the externally obtained
checkpoint.

.. _notices: https://github.com/sickrichmond/AttackBench/blob/main/THIRD_PARTY_NOTICES.md
.. _stutz_repo: https://github.com/davidstutz/confidence-calibrated-adversarial-training#license
.. _xiao_release: https://github.com/wielandbrendel/robustness_workshop/releases/tag/v0.0.1
