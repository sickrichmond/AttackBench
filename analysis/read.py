import json
import warnings
from pathlib import Path
from typing import Mapping, Tuple, Union

import numpy as np

from utils import Scenario, get_model_key
import sys
from pathlib import Path
# Add parent dir to path to allow importing from attackbench package
sys.path.append(str(Path(__file__).resolve().parent.parent))
from attackbench.wandb_utils import get_precompiled_distances


def read_distances(info_file: Union[Path, str],
                   distance_type: str = 'best',
                   already_adv_distance: float = 0,
                   worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    threat_model = config['attack']['threat_model']
    model = get_model_key(config['model']['name'])


    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get hashes and distances for the adversarial examples
    hashes = info['hashes']
    distances = np.array(info['best_optim_distances' if distance_type == 'best' else 'distances'][threat_model])
    ori_success = np.array(info['ori_success'])
    adv_success = np.array(info['adv_success'])

    # check that adversarial examples have 0 distance for adversarial clean samples
    if (n := np.count_nonzero(distances[ori_success])):
        warnings.warn(f'{n} already adversarial clean samples have non zero perturbations in {info_file}.')

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    distances[~adv_success] = worst_case_distance
    distances[ori_success] = already_adv_distance

    # store results
    #scenario = Scenario(dataset=dataset, batch_size=batch_size, attack=attack, library=lib, threat_model=threat_model,
    #                    model=model)
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    hash_distances = {hash: distance for (hash, distance) in zip(hashes, distances)}
    return scenario, hash_distances


def read_info(info_file: Union[Path, str],
                 already_adv_distance: float = 0,
                 worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['model']['dataset']
    batch_size = str(config['dataset']['batch_size'])
    model = get_model_key(config['model']['name'])
    threat_model = config['attack']['threat_model']

    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get distances for the adversarial examples wrt the given threat model
    for key in info.keys():
        if isinstance(info[key], list):
            info[key] = np.array(info[key])

    ori_success = info['ori_success']
    adv_success = info['adv_success']

    # check that adversarial examples have 0 distance for adversarial clean samples
    #if (n := np.count_nonzero(info['distances'][info['ori_success']])):
    #    warnings.warn(f'{n} already adversarial clean samples have non zero perturbations in {info_file}.')

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    for distance_type in ['distances', 'best_optim_distances']:
        info[distance_type] = np.array(info[distance_type][threat_model])
        info[distance_type][~adv_success] = worst_case_distance
        info[distance_type][ori_success] = already_adv_distance
    info['adv_valid_success'] = adv_success & (~ori_success)

    # store results
    scenario = Scenario(dataset=dataset, batch_size=batch_size, threat_model=threat_model, model=model)
    return scenario, info

def load_best_distances(scenario, result_path):
    """
    Helper to load best distances using W&B utils (handles cache/download).
    """
    # Try to download/load from cache
    data = get_precompiled_distances(
        dataset=scenario.dataset,
        threat_model=scenario.threat_model,
        model_name=scenario.model,
        batch_size=scenario.batch_size,
        cache_dir=result_path # Use results folder as cache
    )

    # Fallback for local legacy files not on W&B
    if data is None:
        local_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}.json'
        if local_file.exists():
            with open(local_file, 'r') as f:
                return json.load(f)
        return {}

    return data