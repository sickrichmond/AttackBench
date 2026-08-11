"""
Best-of-MinNorm (BoMN) composite attack.

Runs a set of attacks under the same benchmark conditions and keeps, for each sample,
the smallest perturbation any of them found. This is the empirical best attack a* of the
AttackBench paper, restricted to the attacks you pass in.

Every attack goes through run_attack(), so BoMN inherits the query budget, the d*
tracking and the per-sample hashing, and its output has the same shape as a single run —
it can be fed to get_stats() or uploaded as a lower envelope like any other result.
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..run import run_attack, DEFAULT_QUERY_BUDGET


def _attack_label(attack: Callable, lib: Optional[str] = None) -> str:
    """Readable 'name-library' label, using the metadata run_attack also relies on."""
    func = getattr(attack, 'func', attack)
    name = (getattr(attack, '_attackbench_name', None)
            or getattr(func, '_attackbench_name', None)
            or getattr(func, '__name__', 'unknown'))
    lib = (lib or getattr(attack, '_attackbench_lib', None)
           or getattr(func, '_attackbench_lib', None) or 'unknown')
    return f'{name}-{lib}'


def bomn_attack(
    model: nn.Module,
    dataset: DataLoader,
    attacks: List[Callable],
    threat_model: str,
    attack_libs: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    query_budget: Optional[int] = DEFAULT_QUERY_BUDGET,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Best-of-MinNorm: run every attack, keep the best perturbation per sample.

    Args:
        model: PyTorch model to attack
        dataset: DataLoader with inputs to attack
        attacks: Attack callables (e.g. [pgd, apgd, fmn])
        threat_model: Norm the selection is made on ('l0', 'l1', 'l2', 'linf')
        attack_libs: Optional library name per attack (otherwise auto-detected)
        device: Device to run on
        query_budget: Per-attack query budget; each attack gets the full budget, exactly
            as when they are benchmarked individually
        verbose: Print a per-attack summary
        **kwargs: Forwarded to run_attack / the attacks

    Returns:
        Same schema as run_attack() — distances (the envelope), final_distances,
        adv_success, ori_success, correct, hashes, query counts, metadata — plus:
        - best_attack_indices: index of the winning attack per sample (-1 if all failed)
        - attack_names: labels of the attacks, in the order they were passed
    """
    if not attacks:
        raise ValueError('bomn_attack needs at least one attack')
    if attack_libs is not None and len(attack_libs) != len(attacks):
        raise ValueError(f'attack_libs ({len(attack_libs)}) must match attacks ({len(attacks)})')

    libs = attack_libs or [None] * len(attacks)
    attack_names = [_attack_label(a, lib) for a, lib in zip(attacks, libs)]

    results = []
    for name, attack in zip(attack_names, attacks):
        if verbose:
            print(f'[BoMN] running {name}')
        results.append(run_attack(model, dataset, attack, threat_model, device=device,
                                  query_budget=query_budget, **kwargs))

    if threat_model not in results[0]['distances']:
        raise ValueError(f"No distances for threat model '{threat_model}'")

    hashes = results[0]['hashes']
    if any(r['hashes'] != hashes for r in results):
        raise ValueError('The attacks were evaluated on different samples (hash mismatch)')

    # (n_attacks, n_samples): the norm the selection is made on
    selection = np.array([r['distances'][threat_model] for r in results], dtype=float)
    winner = selection.argmin(axis=0)
    best = selection.min(axis=0)

    # The winning adversarial example defines every other quantity for that sample
    def gather(key: str, norm: str = None):
        values = np.array([(r[key][norm] if norm else r[key]) for r in results])
        return values[winner, np.arange(values.shape[1])].tolist()

    distances = {norm: gather('distances', norm) for norm in results[0]['distances']}
    final_distances = {norm: gather('final_distances', norm) for norm in results[0]['final_distances']}

    meta = dict(results[0].get('metadata', {}))
    meta.update({'attack_name': 'bomn', 'attack_lib': 'attackbench', 'source': 'bomn',
                 'components': attack_names})

    composite = {
        'distances': distances,
        'final_distances': final_distances,
        'adv_success': gather('adv_success'),
        'ori_success': results[0]['ori_success'],
        'correct': results[0]['correct'],
        'hashes': hashes,
        # The composite spends what all of its components spent
        'num_forwards': np.sum([r['num_forwards'] for r in results], axis=0).tolist(),
        'num_backwards': np.sum([r['num_backwards'] for r in results], axis=0).tolist(),
        'times': [sum(t) for t in zip(*[r['times'] for r in results])],
        'box_failures': np.any([r['box_failures'] for r in results], axis=0).tolist(),
        'batch_failures': np.any([r['batch_failures'] for r in results], axis=0).tolist(),
        'targeted': results[0]['targeted'],
        'query_budget': query_budget,
        'best_attack_indices': np.where(np.isfinite(best), winner, -1).tolist(),
        'attack_names': attack_names,
        'metadata': meta,
    }

    if verbose:
        solved = np.isfinite(best)
        print(f'\nBoMN-{threat_model.upper()} over {len(attacks)} attacks, '
              f'{len(best)} samples: {solved.sum()} solved')
        for i, name in enumerate(attack_names):
            wins = int(((winner == i) & solved).sum())
            print(f'  {name:<30} best on {wins:>5} samples')
        if solved.any():
            print(f'  mean distance (solved): {best[solved].mean():.6f}')

    return composite
