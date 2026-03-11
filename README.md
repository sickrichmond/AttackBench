# **AttackBench**: Evaluating Gradient-based Attacks for Adversarial Examples

Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli, and Riccardo Trebiani

**Leaderboard**: [https://attackbench.github.io/](https://attackbench.github.io/)

**Paper:** [https://arxiv.org/pdf/2404.19460](https://arxiv.org/pdf/2404.19460)

**Tutorial Notebook:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rzzLRjMovcns25qOeEXt15R3L2Md_Pst?usp=sharing)

## How it works

The <code>AttackBench</code> framework wants to fairly compare gradient-based attacks based on their security evaluation curves. To this end, we derive a process involving five distinct stages, as depicted below.
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

- Python >= 3.9, < 3.13
- PyTorch >= 2.4
- TorchVision >= 0.19
- CUDA compatible GPU (recommended)

### Install from PyPI

```bash
pip install attackbench
```

### Optional dependencies

```bash
# Attack library wrappers (ART, Foolbox, Torchattacks, CleverHans, RobustBench)
pip install "attackbench[attacks]"

# Model loading utilities (RobustBench, timm, transformers)
pip install "attackbench[models]"

# Analysis and visualization tools (scikit-learn, seaborn, plotly)
pip install "attackbench[metrics]"

# Everything (attacks + models + metrics)
pip install "attackbench[all]"
```

> **Note:** `adv-lib` is not on PyPI. Install it manually if needed:
> `pip install git+https://github.com/jeromerony/adversarial-library`
>
> `deeprobust` requires `scipy<1.8.0` and only works on Python 3.9:
> `pip install "attackbench[deeprobust]"`

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

# Load model and dataset
model = attackbench.get_model('Standard')
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

# Analyze results (requires attackbench[metrics])
stats = attackbench.get_stats(results, 'linf')
print(f"ASR: {stats['asr']*100:.1f}%")
```

Preconfigured attacks available out of the box: `pgd`, `fgsm`, `apgd`, `fab`, `fmn`, `deepfool`, `superdeepfool`, `trust_region`.

To use attacks from external libraries (requires `attackbench[attacks]`):

```python
# List available attacks
attacks = attackbench.list_attacks(threat_model='linf')

# Load a specific library attack
art_pgd = attackbench.get_attack(lib='art', attack='pgd', threat_model='linf')
results = attackbench.run_attack(model=model, dataset=dataset, attack=art_pgd, threat_model='linf', device=device)
```



## Attack format

Tthe wrappers for all the implementations (including libraries) must have the following format:

- inputs:
    - `model`: `nn.Module` taking inputs in the [0, 1] range and returning logits in $\mathbb{R}^K$
    - `inputs`: `FloatTensor` representing the input samples in the [0, 1] range
    - `labels`: `LongTensor` representing the labels of the samples
    - `targets`: `LongTensor` or `None` representing the targets associated to each samples
    - `targeted`: `bool` flag indicating if a targeted attack should be performed
- output:
    - `adv_inputs`: `FloatTensor` representing the perturbed inputs in the [0, 1] range


## Citation

If you use the **AttackBench** leaderboards or implementation, then consider citing our [paper]():

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
