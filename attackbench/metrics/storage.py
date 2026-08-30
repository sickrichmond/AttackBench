"""
Local storage of precompiled distances, in the layout expected by the W&B artifacts.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def save_precompiled_distances(
    attack_data: Dict[str, Any],
    output_dir: str = "./temp_upload",
    dataset: Optional[str] = None,
    threat_model: Optional[str] = None,
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    attack_lib: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Save precompiled distances to a local JSON file with the canonical artifact name.

    Metadata is taken from attack_data['metadata'] (populated by run_attack); pass any
    of the fields explicitly to override it.

    Returns:
        tuple: (file_path, metadata_dict) — ready for upload_precompiled_distances()
    """
    meta = attack_data.get("metadata", {})
    dataset = dataset or meta.get("dataset")
    threat_model = threat_model or meta.get("threat_model")
    model_name = model_name or meta.get("model_name")
    attack_name = attack_name or meta.get("attack_name")
    attack_lib = attack_lib or meta.get("attack_lib")

    missing = [
        n
        for n, v in [
            ("dataset", dataset),
            ("threat_model", threat_model),
            ("model_name", model_name),
            ("attack_name", attack_name),
            ("attack_lib", attack_lib),
        ]
        if not v
    ]
    if missing:
        raise ValueError(
            f"Missing metadata {missing}: pass it explicitly or run the attack "
            f"through run_attack(), which records it in results['metadata']."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Include the basic metrics (ASR, accuracy) alongside the raw data
    from .distances import compute_basic_metrics

    n_samples = len(attack_data.get("adv_success", []))
    metadata = {
        "dataset": dataset,
        "model_name": model_name,
        "attack_name": attack_name,
        "attack_lib": attack_lib,
        "threat_model": threat_model,
        "n_samples": n_samples,
        "query_budget": attack_data.get("query_budget"),
        "protocol_version": 2,
        "distance_semantics": "best_observed",
    }
    save_data = {
        "distances": attack_data.get("distances", {}),
        "final_distances": attack_data.get("final_distances", {}),
        "adv_success": attack_data.get("adv_success", []),
        "ori_success": attack_data.get("ori_success", []),
        "correct": attack_data.get("correct", []),
        "hashes": attack_data.get("hashes", []),
        "original_predictions": attack_data.get("original_predictions", []),
        "adversarial_predictions": attack_data.get("adversarial_predictions", []),
        "num_forwards": attack_data.get("num_forwards", []),
        "num_backwards": attack_data.get("num_backwards", []),
        "times": attack_data.get("times", []),
        "box_failures": attack_data.get("box_failures", []),
        "batch_failures": attack_data.get("batch_failures", []),
        "targeted": attack_data.get("targeted", False),
        "query_budget": attack_data.get("query_budget"),
        "threat_model": threat_model,
        "metadata": metadata,
        **compute_basic_metrics(attack_data),
    }

    # Filename: dataset-threat_model-model-attack-lib-nsamples.json
    filename = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}.json"
    file_path = output_path / filename

    with open(file_path, "w") as f:
        json.dump(save_data, f, indent=2, default=_json_default)

    return str(file_path), metadata


def _json_default(value):
    """numpy scalars/arrays are not JSON serializable on their own."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_precompiled_distances(file_path: str) -> Optional[Dict[str, Any]]:
    """Load precompiled distances from file."""
    path = Path(file_path)
    if not path.exists():
        return None

    try:
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            return {
                key: (data[key].item() if data[key].ndim == 0 else data[key])
                for key in data.files
            }
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
