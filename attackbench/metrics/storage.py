"""
Storage utilities with W&B integration for precompiled distances.
Now uses the new wandb_manager module.
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch


def save_precompiled_distances(
    attack_data: Dict[str, Any], 
    dataset: str,
    threat_model: str, 
    model_name: str,
    attack_name: str,
    attack_lib: str,
    output_dir: str = './temp_upload',
    format: str = 'json'
) -> tuple[str, Dict[str, Any]]:
    """
    Save precompiled distances to local file with automatic naming.
    
    Args:
        attack_data: Raw attack results dict
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf')
        model_name: Model name (e.g., 'Standard')
        attack_name: Attack name (e.g., 'pgd')
        attack_lib: Library implementing the attack (e.g., 'foolbox', 'torchattacks')
        output_dir: Output directory
        format: Output format ('json')
    
    Returns:
        tuple: (file_path, metadata_dict) for easy upload
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for saving (include basic metrics now computed here)
    from .distances import compute_basic_metrics
    basic_metrics = compute_basic_metrics(attack_data)
    
    save_data = {
        'distances': attack_data.get('distances', {}),
        'best_optim_distances': attack_data.get('best_optim_distances', {}),
        'adv_success': attack_data.get('adv_success', []),
        'ori_success': attack_data.get('ori_success', []),
        'hashes': attack_data.get('hashes', []),
        'threat_model': threat_model,
        # Add computed metrics
        **basic_metrics
    }
    
    # Add metadata
    n_samples = len(attack_data.get('adv_success', []))
    metadata = {
        'dataset': dataset,
        'model_name': model_name,
        'attack_name': attack_name,
        'attack_lib': attack_lib,
        'threat_model': threat_model,
        'n_samples': n_samples
    }
    save_data['metadata'] = metadata
    
    # Create filename: dataset-threat_model-model-attack-lib-nsamples.json
    filename = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}.json"
    file_path = output_path / filename
    
    # Convert numpy arrays to lists for JSON
    json_data = {}
    for key, value in save_data.items():
        if isinstance(value, np.ndarray):
            json_data[key] = value.tolist()
        elif isinstance(value, dict):
            json_data[key] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                            for k, v in value.items()}
        else:
            json_data[key] = value
    
    with open(file_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return str(file_path), metadata


def load_precompiled_distances(file_path: str) -> Optional[Dict[str, Any]]:
    """Load precompiled distances from file."""
    path = Path(file_path)
    if not path.exists():
        return None
    
    try:
        if path.suffix == '.npz':
            data = np.load(path, allow_pickle=True)
            result = {}
            for key in data.files:
                result[key] = data[key].item() if data[key].ndim == 0 else data[key]
            return result
        else:
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_best_distances_with_wandb(dataset: str, threat_model: str, model_name: str, 
                                  batch_size: int, cache_dir: str) -> Dict[str, Any]:
    """
    Load best distances using new W&B manager.
    """
    try:
        # Use the new modular function
        from ..wandb.manager import download_precompiled_distances
        
        data = download_precompiled_distances(
            dataset=dataset,
            threat_model=threat_model,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir
        )
        
        return data or {}
        
    except ImportError:
        print("W&B manager not available, trying legacy fallback")
        # Fallback al vecchio sistema se necessario
        try:
            from ..wandb.utils import get_precompiled_distances
            return get_precompiled_distances(dataset, threat_model, model_name, batch_size, cache_dir) or {}
        except ImportError:
            print("No W&B integration available")
    
    return {}