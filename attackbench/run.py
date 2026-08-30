import hashlib
import inspect
import json
import random
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

# W&B integration for cached distances
from .wandb.manager import PRECOMPILED_RESULT_FIELDS, download_precompiled_distances

# Max forward+backward propagations per sample. 2000 is the budget used in the
# AttackBench paper: it is what makes attacks comparable to one another.
DEFAULT_QUERY_BUDGET = 2000

# Fields persisted in precompiled artifacts. A cache hit must be indistinguishable
# from an executed run for downstream consumers such as BoMN.
_CACHED_RESULT_FIELDS = PRECOMPILED_RESULT_FIELDS


def _hash_batch(inputs: Tensor) -> list:
    """SHA-512 of each sample, to check that runs and cached results share the samples."""
    return [hashlib.sha512(np.ascontiguousarray(x.numpy())).hexdigest() for x in inputs]


def _set_seed(seed: int = None) -> None:
    """Set random seed for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _call_attack(attack, model, inputs, labels, targeted, targets, **kwargs):
    """
    Call the attack function, filtering kwargs to match its signature.
    """
    sig = inspect.signature(attack)

    attack_params = {
        "model": model,
        "inputs": inputs,
        "labels": labels,
    }

    if "targeted" in sig.parameters:
        attack_params["targeted"] = targeted
    if "targets" in sig.parameters:
        attack_params["targets"] = targets

    # AttackBench historically exposed the requested norm to custom/library wrappers
    # as ``norm``. Preconfigured attacks use the clearer ``threat_model`` parameter;
    # route the same value to that explicit parameter without injecting a new keyword
    # into arbitrary third-party ``**kwargs`` wrappers.
    if "threat_model" in sig.parameters and "norm" in kwargs:
        attack_params["threat_model"] = kwargs["norm"]

    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in sig.parameters.values()
    )
    for key, value in kwargs.items():
        if key == "norm" and "threat_model" in sig.parameters:
            continue
        if accepts_kwargs or key in sig.parameters:
            attack_params[key] = value

    return attack(**attack_params)


def _extract_metadata(model, dataset, attack):
    """
    Extract metadata from objects for W&B caching.

    Returns:
        tuple: (dataset_name, model_name, attack_name, attack_lib)
               Any value may be None if extraction fails.
    """
    # Extract dataset name from DataLoader (set by get_loader)
    dataset_name = getattr(dataset, "_attackbench_dataset", None)

    # Extract model name from model (set by attackbench.load_model / get_model)
    model_name = getattr(model, "_attackbench_model", None)
    # Fallback: try model.model for BenchModel-wrapped models
    if model_name is None and hasattr(model, "model"):
        model_name = getattr(model.model, "_attackbench_model", None)

    # Fallback: read dataset from model if loader didn't provide it
    # (set by attackbench.load_model which receives dataset as parameter)
    if dataset_name is None:
        dataset_name = getattr(model, "_attackbench_dataset", None)
        if dataset_name is None and hasattr(model, "model"):
            dataset_name = getattr(model.model, "_attackbench_dataset", None)

    # Extract attack name and lib from attack callable
    attack_name = getattr(attack, "_attackbench_name", None)
    attack_lib = getattr(attack, "_attackbench_lib", None)

    # Fallback: try to get name from partial/function
    if attack_name is None:
        if hasattr(attack, "func"):  # functools.partial
            attack_name = getattr(attack.func, "__name__", None)
        else:
            attack_name = getattr(attack, "__name__", None)

    # Fallback: try to infer library from module
    if attack_lib is None:
        module = None
        if hasattr(attack, "func"):
            module = getattr(attack.func, "__module__", "")
        else:
            module = getattr(attack, "__module__", "")

        if module:
            # Extract library name from module path
            # e.g., 'attack_evaluation.attacks.foolbox.wrapper' -> 'foolbox'
            parts = module.split(".")
            for lib_name in [
                "foolbox",
                "torchattacks",
                "adv_lib",
                "art",
                "cleverhans",
                "deeprobust",
                "original",
            ]:
                if lib_name in parts:
                    attack_lib = lib_name
                    break

    return dataset_name, model_name, attack_name, attack_lib


def run_attack(
    model: nn.Module,
    dataset: DataLoader,
    attack: Callable,
    threat_model: str,
    device: Optional[torch.device] = None,
    query_budget: Optional[int] = DEFAULT_QUERY_BUDGET,
    save_results: bool = False,
    save_adversarial: bool = False,
    output_dir: Optional[str] = None,
    seed: int = 42,
    debug: bool = False,
    # Metadata for W&B integration and optimality computation
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    attack_lib: Optional[str] = None,
    # Caching options
    use_cached: bool = False,
    cache_dir: str = "./cache",
    **kwargs,
) -> Dict[str, Any]:
    """
    Execute an adversarial attack and return raw benchmark results.

    For derived analysis and statistics, use attackbench.get_stats() afterwards.

    **Automatic Metadata Extraction:**
    When using get_loader(), get_model(), and get_attack() helpers, metadata
    (dataset_name, model_name, attack_name, attack_lib) is automatically extracted
    from the objects. You don't need to specify these parameters manually.

    **Query budget:**
    By default every attack is limited to DEFAULT_QUERY_BUDGET (2000) forward+backward
    propagations per sample, the budget used in the AttackBench paper — that limit is
    what makes attacks comparable to each other. Pass query_budget=None to run without
    a budget (debugging, exploratory runs); the results are then NOT comparable with
    the published leaderboard.

    Args:
        model: PyTorch model to attack
        dataset: DataLoader with inputs to attack
        attack: Attack function/callable
        threat_model: Threat model ('linf', 'l2', 'l1', 'l0')
        device: Device to run on (default: auto-detect)
        query_budget: Max forward+backward propagations per sample (None = unlimited).
            Overrides a budget already set on a BenchModel by get_model().
        save_results: Save results to disk
        save_adversarial: Save adversarial examples
        output_dir: Directory for saved results
        seed: Random seed
        debug: Debug mode — let attack exceptions propagate instead of recording a failure
        dataset_name: Override auto-detected dataset name
        model_name: Override auto-detected model name
        attack_name: Override auto-detected attack name
        attack_lib: Override auto-detected attack library
        use_cached: If True, look up precompiled distances on W&B before running, and
            reuse them only if their per-sample hashes match this dataset exactly.
            Off by default: a benchmark should measure the attack you passed it.
        cache_dir: Directory for local cache of W&B downloads
        **kwargs: Additional arguments passed to attack

    Returns:
        Dict with MINIMAL RAW attack data:
        - distances: Dict[str, List[float]] — d*, the smallest perturbation found
          during the optimization (AttackBench Alg. 1), per norm. This is what the
          optimality metric is computed on. 0 for already-misclassified samples,
          inf for samples the attack never broke.
        - final_distances: Dict[str, List[float]] — distance of the sample the attack
          actually returned (last iterate). Diagnostics only: >= distances by
          construction, and a large gap means the attack discards its own best result.
        - adv_success: List[bool] (attack success per sample - needed for ASR)
        - ori_success: List[bool] (sample was ALREADY misclassified before the attack)
        - correct: List[bool] (sample was correctly classified before the attack;
          clean accuracy is mean(correct))

        - hashes: List[str] (SHA-512 per sample, for hash-based matching)

        Always included — these are the failure indicators the benchmark exists to
        surface, so they are not hidden behind a flag:
        - num_forwards / num_backwards: List[int] (query counts per sample)
        - times: List[float] (per batch)
        - box_failures: List[bool] (attack produced values outside [0, 1])
        - batch_failures: List[bool] (the attack raised on that batch)
        - original_predictions / adversarial_predictions: List[int]
        - targeted: bool

        If save_adversarial=True, also includes:
        - adv_inputs: Tensor (adversarial examples)
        - inputs: Tensor (original inputs)
    """

    # Input validation
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be nn.Module, got {type(model)}")

    if not isinstance(dataset, DataLoader):
        raise TypeError(f"dataset must be DataLoader, got {type(dataset)}")

    if not callable(attack):
        raise TypeError(f"attack must be callable, got {type(attack)}")

    has_attack_overrides = bool(kwargs)
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    _set_seed(seed)

    # Wrap model if necessary
    if not hasattr(model, "start_tracking"):
        from .models.benchmodel_wrapper import BenchModel

        model = BenchModel(model)

    # The query budget is the benchmark's fairness guarantee, so run_attack owns it:
    # an explicit budget here wins over whatever get_model() set on the BenchModel.
    model.num_max_propagations = query_budget

    model.to(device)

    # Add norm to kwargs for custom attacks
    kwargs["norm"] = threat_model

    if len(dataset) == 0:
        raise ValueError("Dataset is empty - no inputs to attack")

    # Auto-extract metadata from objects if not provided
    auto_dataset, auto_model, auto_attack, auto_lib = _extract_metadata(
        model, dataset, attack
    )
    dataset_name = dataset_name or auto_dataset
    model_name = model_name or auto_model
    attack_name = attack_name or auto_attack
    attack_lib = attack_lib or auto_lib

    # Precompiled artifacts contain the canonical named attack, not arbitrary runtime
    # overrides, and cannot supply tensors requested by save_adversarial.
    if use_cached and has_attack_overrides:
        warnings.warn(
            "Ignoring use_cached=True because attack keyword overrides were supplied; "
            "the artifact key does not encode those hyperparameters."
        )
        use_cached = False
    if use_cached and save_adversarial:
        warnings.warn(
            "Ignoring use_cached=True because precompiled artifacts do not contain "
            "adversarial input tensors."
        )
        use_cached = False

    # Check for cached precompiled distances on W&B. The artifact name only encodes
    # (dataset, norm, model, attack, lib, n_samples), which does not pin down WHICH samples
    # were used nor the attack's hyperparameters — so a hit is only trusted when the
    # per-sample hashes match. Computing them costs a pass over the data, hence only here.
    if use_cached and all([dataset_name, model_name, attack_name, attack_lib]):
        expected_hashes = [h for inputs, _ in dataset for h in _hash_batch(inputs)]
        n_samples = len(expected_hashes)
        key = f"{dataset_name}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}".lower()
        print(f"[AttackBench] Checking W&B for cached distances: {key}")

        cached_data = download_precompiled_distances(
            dataset=dataset_name,
            threat_model=threat_model,
            model_name=model_name,
            attack_name=attack_name,
            attack_lib=attack_lib,
            n_samples=n_samples,
            cache_dir=cache_dir,
        )

        if cached_data is not None and cached_data.get("hashes") != expected_hashes:
            warnings.warn(
                "Cached distances found on W&B, but their per-sample hashes do not match this "
                "dataset (different subset, seed or preprocessing). Ignoring the cache and "
                "running the attack."
            )
            cached_data = None

        if cached_data is not None:
            missing = [key for key in _CACHED_RESULT_FIELDS if key not in cached_data]
            if missing:
                warnings.warn(
                    f"Cached artifact uses an incomplete pre-2.0 schema (missing "
                    f"{missing}). Ignoring it and running the attack."
                )
                cached_data = None

        if cached_data is not None and cached_data["query_budget"] != query_budget:
            warnings.warn(
                f'Cached artifact used query_budget={cached_data["query_budget"]!r}, '
                f"but this run requested {query_budget!r}. Ignoring the cache."
            )
            cached_data = None

        if cached_data is not None:
            print(
                f"[AttackBench] Reusing cached distances (hashes match); skipping execution."
            )
            result = {key: cached_data[key] for key in _CACHED_RESULT_FIELDS}
            result["hashes"] = expected_hashes
            result["metadata"] = {
                **cached_data.get("metadata", {}),
                "dataset": dataset_name,
                "model_name": model_name,
                "attack_name": attack_name,
                "attack_lib": attack_lib,
                "threat_model": threat_model,
                "n_samples": n_samples,
                "query_budget": query_budget,
                "protocol_version": 2,
                "distance_semantics": "best_observed",
                "source": "wandb_cache",
            }
            return result
        print(f"[AttackBench] No usable cached distances. Running attack...")

    # ── Execute attack (batch loop) ──────────────────────────────────────
    from .adv_lib_sub import _default_metrics

    metrics = _default_metrics

    targeted = False
    loader_length = len(dataset)

    correct, ori_success, adv_success = [], [], []
    hashes_list, box_failures, batch_failures = [], [], []
    predictions, adv_predictions = [], []
    forwards, backwards, times = [], [], []
    distances, final_distances = defaultdict(list), defaultdict(list)

    if save_adversarial:
        all_inputs, all_adv_inputs = [], []

    for inputs, labels in tqdm(dataset, ncols=80, total=loader_length):
        if save_adversarial:
            all_inputs.append(inputs.clone())

        # Compute hashes to ensure input samples are identical
        hashes_list.extend(_hash_batch(inputs))

        inputs, labels = inputs.to(device), labels.to(device)
        attack_inputs, attack_labels = inputs.clone(), labels.clone()

        # Start tracking of the batch
        model.start_tracking(
            inputs=inputs,
            labels=labels,
            targeted=targeted,
            targets=None,
            tracking_metric=metrics[threat_model],
            tracking_threat_model=threat_model,
        )

        if debug:
            adv_inputs = _call_attack(
                attack,
                model,
                attack_inputs,
                attack_labels,
                targeted,
                None,
                threat_model=threat_model,
                **kwargs,
            )
            batch_failures.append(False)
        else:
            try:
                adv_inputs = _call_attack(
                    attack,
                    model,
                    attack_inputs,
                    attack_labels,
                    targeted,
                    None,
                    threat_model=threat_model,
                    **kwargs,
                )
                batch_failures.append(False)
            except Exception:
                warnings.warn(f"Error running batch for {attack}")
                traceback.print_exc()
                batch_failures.append(True)
                adv_inputs = inputs

        model.end_tracking()
        adv_inputs.detach_()
        times.append(model.elapsed_time)
        forwards.extend(model.num_forwards.cpu().tolist())
        backwards.extend(model.num_backwards.cpu().tolist())

        # Original inputs
        correct.extend(model.correct.cpu().tolist())
        ori_success.extend(model.ori_success.cpu().tolist())

        # Checking box constraint
        batch_box_failures = ((adv_inputs < 0) | (adv_inputs > 1)).flatten(1).any(1)
        box_failures.extend(batch_box_failures.cpu().tolist())

        if batch_box_failures.any():
            warnings.warn(
                "Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1]."
            )
            adv_inputs.clamp_(min=0, max=1)

        if save_adversarial:
            all_adv_inputs.append(adv_inputs.cpu().clone())

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)

        ori_logits = model(inputs)
        ori_pred = ori_logits.argmax(dim=1)
        predictions.extend(ori_pred.cpu().tolist())
        adv_predictions.extend(adv_pred.cpu().tolist())

        success = adv_pred != labels
        adv_success.extend(success.cpu().tolist())

        for metric_name, metric_func in metrics.items():
            final = metric_func(adv_inputs, inputs)
            # Failed attacks return the original input (distance ≈ 0).
            # Set distance to inf so they are correctly treated as failures
            # in SEC/optimality computations. Preserve distance=0 for samples
            # that were already misclassified (ori_success=True).
            final[~success & ~model.ori_success] = float("inf")
            # d* (AttackBench Alg. 1): the best perturbation found *during* the
            # optimization, not the last iterate. min_dist only updates on tracked
            # queries, so an attack returning a better sample without re-querying it
            # would be underestimated => take the elementwise minimum of the two.
            best = torch.minimum(model.min_dist[metric_name], final)
            distances[metric_name].extend(best.cpu().tolist())
            final_distances[metric_name].extend(final.cpu().tolist())

    # ── Package results ──────────────────────────────────────────────────
    n_samples = len(adv_success)

    clean_data = {
        "distances": dict(distances),  # d*: best found during optimization
        "final_distances": dict(final_distances),  # last iterate, for diagnostics only
        "adv_success": adv_success,
        "ori_success": ori_success,
        "correct": correct,  # clean correctness per sample (accuracy is mean(correct))
        "hashes": hashes_list,  # Always included for sample identity tracking
        # Failure indicators and query counts: always returned. These are what tells a
        # broken attack implementation from a strong model, which is the whole point of
        # the benchmark — gating them behind a flag hides exactly what matters.
        "original_predictions": predictions,
        "adversarial_predictions": adv_predictions,
        "num_forwards": forwards,
        "num_backwards": backwards,
        "times": times,
        "box_failures": box_failures,
        "batch_failures": batch_failures,
        "targeted": targeted,
        "query_budget": query_budget,
        "metadata": {
            "dataset": dataset_name,
            "model_name": model_name,
            "attack_name": attack_name,
            "attack_lib": attack_lib,
            "threat_model": threat_model,
            "n_samples": n_samples,
            "query_budget": query_budget,
            "protocol_version": 2,
            "distance_semantics": "best_observed",
            "source": "executed",
        },
    }

    # Add adversarial inputs if requested
    if save_adversarial:
        clean_data["inputs"] = torch.cat(all_inputs, dim=0)
        clean_data["adv_inputs"] = torch.cat(all_adv_inputs, dim=0)

    # Save raw results if requested
    if save_results or save_adversarial:
        if output_dir:
            # Simple flat directory structure
            save_dir = Path(output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Save adversarial data
            if save_adversarial and "adv_inputs" in clean_data:
                torch.save(clean_data, save_dir / "attack_data.pt")

            # Save results JSON (without large tensors)
            results_to_save = {
                k: v for k, v in clean_data.items() if k not in ["inputs", "adv_inputs"]
            }

            with open(save_dir / "results.json", "w") as f:
                json.dump(results_to_save, f, indent=2, default=str)

    return clean_data
