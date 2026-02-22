import wandb
import json
from pathlib import Path

# Configuration constants
WANDB_ENTITY = "attackbench"
WANDB_PROJECT = "attackbench-precompiled-distancies"
WANDB_PROJECT_OPTIMAL = "attackbench-optimal-distancies"

def get_precompiled_distances(dataset, threat_model, model_name, attack_name, attack_lib, n_samples, cache_dir="./data/cache"):
    """
    Retrieves precompiled distances for a specific attack. Checks local cache first, then downloads from W&B.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        attack_name: Attack name (e.g., 'pgd', 'apgd')
        attack_lib: Library implementing the attack (e.g., 'foolbox', 'torchattacks', 'adv_lib')
        n_samples: Number of samples
        cache_dir: Local cache directory
        
    Returns:
        Dictionary with precompiled distances, or None if not found
    """
    # Artifact naming convention: dataset-threat_model-model-attack-lib-n_samples
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}"
    file_name = f"{artifact_name}.json"
    
    cache_path = Path(cache_dir) / dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file_path = cache_path / file_name

    # 1. Check local cache
    if local_file_path.exists():
        try:
            with open(local_file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[AttackBench] Corrupted local file: {local_file_path}. Redownloading...")

    # 2. Download from W&B
    print(f"[AttackBench] Downloading artifact from W&B: {artifact_name}...")
    
    # Initialize API (anonymous mode supported if project is public)
    api = wandb.Api()
    
    try:
        artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:latest")
        download_path = artifact.download(root=str(cache_path))
        
        # Locate the downloaded JSON file
        downloaded_file = Path(download_path) / file_name
        if not downloaded_file.exists():
             # Fallback: look for any json in the folder
             jsons = list(Path(download_path).glob("*.json"))
             downloaded_file = jsons[0] if jsons else None

        if downloaded_file and downloaded_file.exists():
            with open(downloaded_file, 'r') as f:
                data = json.load(f)
            return data
        else:
            raise FileNotFoundError("No JSON file found in the downloaded artifact.")

    except Exception as e:
        print(f"[AttackBench] Warning: Failed to download distances for {model_name}. Error: {e}")
        return None


def get_optimal_distances(dataset, threat_model, model_name, n_samples, cache_dir="./data/cache"):
    """
    Retrieves optimal distances (lower envelope). Checks local cache first, then downloads from W&B.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        n_samples: Number of samples
        cache_dir: Local cache directory
        
    Returns:
        Dictionary with optimal distances, or None if not found
    """
    # Artifact naming convention: dataset-threat_model-model-n_samples (NO attack_name)
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{n_samples}"
    file_name = f"{artifact_name}.json"
    
    # Use separate cache folder for optimal distances
    cache_path = Path(cache_dir) / "optimal" / dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file_path = cache_path / file_name

    # 1. Check local cache
    if local_file_path.exists():
        try:
            with open(local_file_path, 'r') as f:
                print(f"[AttackBench] Loading optimal distances from cache: {local_file_path}")
                return json.load(f)
        except json.JSONDecodeError:
            print(f"[AttackBench] Corrupted local file: {local_file_path}. Redownloading...")

    # 2. Download from W&B
    print(f"[AttackBench] Downloading optimal distances from W&B: {artifact_name}...")
    
    api = wandb.Api()
    
    try:
        artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT_OPTIMAL}/{artifact_name}:latest")
        download_path = artifact.download(root=str(cache_path))
        
        # Locate the downloaded JSON file
        downloaded_file = Path(download_path) / file_name
        if not downloaded_file.exists():
            # Fallback: look for any json in the folder
            jsons = list(Path(download_path).glob("*.json"))
            downloaded_file = jsons[0] if jsons else None

        if downloaded_file and downloaded_file.exists():
            with open(downloaded_file, 'r') as f:
                data = json.load(f)
            print(f"[AttackBench] Successfully loaded optimal distances for {model_name}")
            return data
        else:
            raise FileNotFoundError("No JSON file found in the downloaded artifact.")

    except Exception as e:
        print(f"[AttackBench] Warning: Failed to download optimal distances for {model_name}. Error: {e}")
        return None