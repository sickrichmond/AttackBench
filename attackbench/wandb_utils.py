import wandb
import json
from pathlib import Path

# Configuration constants
WANDB_ENTITY = "attackbench"
WANDB_PROJECT = "attackbench-precompiled-distancies"

def get_precompiled_distances(dataset, threat_model, model_name, batch_size, cache_dir="./data/cache"):
    """
    Retrieves precompiled distances. Checks local cache first, then downloads from W&B.
    """
    # Artifact naming convention: dataset-threat-model-batch
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{batch_size}"
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