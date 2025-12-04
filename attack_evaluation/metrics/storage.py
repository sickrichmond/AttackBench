"""
Storage utilities with W&B integration for precompiled distances.
Enhanced with analysis/read.py logic
"""
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch


def save_precompiled_distances(
    attack_data: Dict[str, Any], 
    threat_model: str, 
    output_dir: str,
    model_name: Optional[str] = None,
    attack_name: Optional[str] = None,
    format: str = 'json'  # Changed default to json
) -> str:
    """Save precompiled distances to local file and optionally W&B."""
    
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
    
    # Add metadata if available
    if model_name or attack_name:
        save_data['metadata'] = {
            'model_name': model_name,
            'attack_name': attack_name,
            'threat_model': threat_model
        }
    
    # Save locally
    if format == 'npz':
        file_path = output_path / 'precompiled_distances.npz'
        np.savez_compressed(file_path, **save_data)
    else:
        file_path = output_path / 'precompiled_distances.json'
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
    
    return str(file_path)


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
    Load best distances using W&B utils. Ported from analysis/read.py
    """
    try:
        from attackbench.wandb_utils import get_precompiled_distances
        
        data = get_precompiled_distances(
            dataset=dataset,
            threat_model=threat_model,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir
        )
        
        if data:
            return data
    except ImportError:
        print("W&B utils not available, trying local fallback")
    
    # Fallback to local file
    local_file = Path(cache_dir) / f'{dataset}-{threat_model}-{model_name}-{batch_size}.json'
    if local_file.exists():
        return load_precompiled_distances(str(local_file)) or {}
    
    return {}