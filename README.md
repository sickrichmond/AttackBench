# **AttackBenchLib**: Evaluating Gradient-based Attacks for Adversarial Examples

Riccardo Trebiani, Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed and Fabio Roli

**Leaderboard**: [https://attackbench.github.io/](https://attackbench.github.io/)

**Paper:** [https://arxiv.org/pdf/2404.19460](https://arxiv.org/pdf/2404.19460)

**Changelog:** [2.0.0 release notes](https://github.com/attackbench/AttackBench/blob/main/CHANGELOG.md)

**Tutorial Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/attackbench/AttackBench/blob/main/examples/AttackBenchLib.ipynb)
## How it works
AttackBenchLib is a library that implements the framework described in the AttackBench paper in a new modular, user-friendly way in order to make multiple workflows and kinds of analysis possible through the use of a single library. 
The <code>AttackBench</code> framework aims to fairly compare gradient-based attacks based on their security evaluation curves. To this end, we derive a process involving five distinct stages, as depicted below.
  - In stage (1), we construct a list of diverse non-robust and robust models to assess the attacks' impact on various settings, thus testing their adaptability to diverse defensive strategies. 
  - In stage (2), we define an environment for testing gradient-based attacks under a systematic and reproducible protocol. 
        This step provides common ground with shared assumptions, advantages, and limitations. 
        We then run the attacks against the selected models individually and collect the performance metrics of interest in our analysis, which are perturbation size, execution time, and query usage. 
  - In stage (3), we gather all the previously-obtained results, comparing  attacks with the novel <code>local optimality</code> metric. 
  - Finally, in stage (4), we aggregate the optimality results from all considered models, and in stage (5) we rank the attacks based on their average optimality, namely <code>global optimality</code>. 
  

<p align="center"><img src="https://attackbench.github.io/assets/AtkBench.svg" width="1300"></p>


## Currently implemented

| Attack       | Original | Advertorch | Adv_lib | ART | CleverHans | DeepRobust | Foolbox | Torchattacks |
|--------------|:--------:|:----------:|:-------:|:---:|:----------:|:----------:|:-------:|:------------:|
| DDN          |    ☒     |            |    ✓    |  ☒  |     ☒      |     ☒      |    ✓    |      ☒       |
| ALMA         |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| FMN          |    ✓     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ✓    |      ☒       |
| PGD          |    ☒     |            |    ✓    |  ✓  |            |     ✓      |         |      ✓       |
| JSMA         |    ☒     |            |    ☒    |  ✓  |     ☒      |     ☒      |    ☒    |      ☒       |
| CW-L2        |    ☒     |            |    ✓    |  ✓  |            |     ~      |    ✓    |      ✓       |
| CW-LINF      |    ☒     |     ☒      |    ✓    |  ✓  |     ☒      |     ☒      |    ☒    |      ☒       |
| FGSM         |    ☒     |            |    ☒    |  ✓  |            |            |         |      ✓       |
| BB           |    ☒     |     ☒      |    ☒    |  ✓  |     ☒      |     ☒      |    ✓    |      ☒       |
| DF           |    ✓     |     ☒      |    ☒    |  ✓  |     ☒      |     ~      |    ✓    |      ✓       |
| SuperDF      |    ✓     |     ☒      |    ☒    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| APGD         |    ✓     |     ☒      |    ✓    |  ✓  |     ☒      |     ☒      |    ☒    |      ✓       |
| BIM          |    ☒     |            |    ☒    |  ✓  |            |     ☒      |         |      ☒       |
| EAD          |    ☒     |            |    ☒    |  ✓  |     ☒      |     ☒      |    ✓    |      ☒       |
| PDGD         |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| PDPGD        |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| TR           |    ✓     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| FAB          |    ✓     |            |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ✓       |


Legend: 
- _empty_ : not implemented yet 
- ☒ : not available
- ✓ : implemented
- ~ : not functional yet



## Requirements and Installation

- Python >= 3.9
- PyTorch >= 2.4
- TorchVision >= 0.19
- CUDA compatible GPU (recommended)

### Install from PyPI

```bash
pip install attackbenchlib
```

### Optional dependencies

```bash
# Attack library wrappers (ART, Foolbox, Torchattacks, CleverHans)
pip install "attackbenchlib[attacks]"

# Model loading utilities (RobustBench)
pip install "attackbenchlib[models]"

# Analysis and evaluation tools
pip install "attackbenchlib[metrics]"

# Everything (attacks + models + metrics)
pip install "attackbenchlib[all]"
```

> **Note on `adv-lib`:** The Adversarial Library (`adv-lib`) is not on PyPI, so it is not
> part of `[all]`. Its implementations are among the top-ranked ones in the AttackBench
> paper, so installing without it leaves you with weaker re-implementations of the same
> attacks. It needs two extra steps: `adv-lib` depends on `visdom`, whose `setup.py`
> imports `pkg_resources`, which setuptools 81+ no longer ships — so `visdom` has to be
> built against an older setuptools first:
> ```bash
> pip install "setuptools<81" wheel
> pip install --no-build-isolation visdom
> pip install "adv-lib @ git+https://github.com/jeromerony/adversarial-library"
> ```
> Without `adv-lib` installed, its attacks are simply absent from `list_attacks()`
> instead of failing when you try to build them.

> **Note on `deeprobust`:** Requires `scipy<1.8.0` and only works on Python 3.9:
> `pip install "attackbenchlib[deeprobust]"`

### Google Colab

On Google Colab, install with all dependencies:

```python
!pip install "attackbenchlib[models,attacks]" -q
```

> Colab's pre-installed packages can conflict with optional attack/model dependencies.
> Review any resolver warning and run `pip check`; if imports fail, restart the runtime
> after installation or use a clean virtual environment.

### Install from source (development)

```bash
git clone https://github.com/attackbench/AttackBench.git
cd AttackBench
pip install -e ".[dev]"
```


## Usage

```python
import torch
import attackbench
from attackbench.attacks import apgd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and dataset (requires attackbenchlib[models])
model = attackbench.load_model('Standard', dataset='cifar10', threat_model='Linf')
model.to(device)

dataset = attackbench.get_loader(dataset='cifar10', batch_size=128, num_samples=1000)

# Run attack
results = attackbench.run_attack(
    model=model,
    dataset=dataset,
    attack=apgd,
    threat_model='linf',
    device=device
)

# Analyze results (requires attackbenchlib[metrics])
stats = attackbench.get_stats(results, 'linf')
print(f"ASR: {stats['ASR']*100:.1f}%")
```

Every attack runs under a budget of 2000 forward+backward propagations per sample — the
budget used in the paper, and what makes attacks comparable to each other. Pass
`run_attack(..., query_budget=None)` to lift it for exploratory runs.

### Upgrading from 1.x

Version 2.0 changes what the numbers mean, so results are not comparable with 1.x:

- `results['distances']` is now `d*`, the smallest perturbation found **during** the
  optimization (as defined in the paper), not the distance of the sample the attack
  returned last. The last iterate is available as `results['final_distances']`.
- `stats['accuracy']` is the clean accuracy. In 1.x it reported the fraction of
  *already misclassified* samples, i.e. the error rate.
- `optimality` is only computed against a real lower envelope — passed in explicitly or
  downloaded from W&B. 1.x silently fell back to the attack's own tracked distances,
  which scored ~1.0 by construction.
- The query budget is enforced by default (see above); 1.x enforced none.
- `run_attack(use_cached=...)` defaults to `False`, and a cached W&B result is only
  reused when its per-sample hashes match the samples being evaluated.
- `include_metadata` is gone: query counts, timings, predictions and the failure
  indicators are always returned.

Preconfigured attacks available with the base installation: `pgd`, `fgsm`, `apgd`, `fab`, `deepfool`, `superdeepfool`, `trust_region`. The preconfigured `fmn` attack additionally requires `attackbenchlib[attacks]`.

To use attacks from external libraries (requires `attackbenchlib[attacks]`):

```python
# List available attacks
attacks = attackbench.list_attacks(threat_model='linf')

# Load a specific library attack
art_pgd = attackbench.get_attack(lib='art', attack='pgd', threat_model='linf')
results = attackbench.run_attack(model=model, dataset=dataset, attack=art_pgd, threat_model='linf', device=device)
```



## Attack format

The wrappers for all the implementations (including libraries) must have the following format:

- inputs:
    - `model`: `nn.Module` taking inputs in the [0, 1] range and returning logits in $\mathbb{R}^K$
    - `inputs`: `FloatTensor` representing the input samples in the [0, 1] range
    - `labels`: `LongTensor` representing the labels of the samples
    - `targets`: `LongTensor` or `None` representing the targets associated to each samples
    - `targeted`: `bool` flag indicating if a targeted attack should be performed
- output:
    - `adv_inputs`: `FloatTensor` representing the perturbed inputs in the [0, 1] range


## Citation

If you use the **AttackBench** leaderboards or implementation, then consider citing our [paper](https://doi.org/10.1609/aaai.v39i3.32263):

```bibtex
@inproceedings{cina2025attackbench,
  title={Attackbench: Evaluating gradient-based attacks for adversarial examples},
  author={Cin{\`a}, Antonio Emanuele and Rony, J{\'e}r{\^o}me and Pintor, Maura and Demetrio, Luca and Demontis, Ambra and Biggio, Battista and Ayed, Ismail Ben and Roli, Fabio},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={3},
  pages={2600--2608},
  year={2025},
  DOI={10.1609/aaai.v39i3.32263}
}
```

## Contact 
Feel free to contact us about anything related to **`AttackBench`** by creating an issue, a pull request or 
by email at `antonio.cina@unige.it`.

## Acknowledgements
AttackBench has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393.

<img src="_static/assets/logos/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/FundedbytheEU.png" alt="europe" style="width:240px;"/>
