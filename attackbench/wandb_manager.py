"""
AttackBench W&B Manager - User-friendly API for precompiled distances.

Provides simple upload/download functions for managing precompiled attack distances.
"""
import json
import os
import wandb
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Configuration constants
WANDB_ENTITY = "attackbench"
WANDB_PROJECT = "attackbench-precompiled-distancies"
WANDB_PROJECT_OPTIMAL = "attackbench-optimal-distancies"

def _get_wandb_api() -> wandb.Api:
    """
    Return a W&B API client.

    Uses the caller's existing credentials (WANDB_API_KEY env var or ~/.netrc)
    when available. Falls back to anonymous access for public projects, so users
    on Colab or fresh environments are never prompted for a login.
    """
    if os.environ.get("WANDB_API_KEY"):
        return wandb.Api()
    # Anonymous access — works when projects are set to Public on wandb.ai
    os.environ.setdefault("WANDB_ANONYMOUS", "allow")
    return wandb.Api(api_key=None)

def upload_precompiled_distances(
    file_path: Union[str, Path] = None,
    attack_data: Dict[str, Any] = None,
    dataset: str = None,
    threat_model: str = None, 
    model_name: str = None,
    attack_name: str = None,
    attack_lib: str = None,
    overwrite: bool = False
) -> bool:
    """
    Upload precompiled distances to W&B.
    
    Two usage modes:
    1. Pass file_path directly (legacy)
    2. Pass attack_data + metadata (automatic file creation)
    
    Args:
        file_path: Path to JSON file (optional if attack_data provided)
        attack_data: Raw attack results dict (optional if file_path provided)
        dataset: Dataset name (required if attack_data provided)
        threat_model: Threat model (required if attack_data provided)
        model_name: Model name (required if attack_data provided)
        attack_name: Attack name (required if attack_data provided)
        attack_lib: Library implementing the attack (e.g., 'foolbox', 'torchattacks')
        overwrite: Whether to overwrite existing artifacts
        
    Returns:
        True if upload successful, False otherwise
    """
    
    # Mode 2: Create file from attack_data
    if attack_data is not None:
        if not all([dataset, threat_model, model_name, attack_name, attack_lib]):
            print("Error: Must provide dataset, threat_model, model_name, attack_name, attack_lib with attack_data")
            return False
        
        from attack_evaluation.metrics.storage import save_precompiled_distances
        file_path, metadata = save_precompiled_distances(
            attack_data, dataset, threat_model, model_name, attack_name, attack_lib
        )
        n_samples = metadata['n_samples']
    else:
        # Mode 1: Use provided file_path
        if file_path is None:
            print("Error: Must provide either file_path or attack_data")
            return False
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return False
        
        # Extract metadata from file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            metadata = data.get('metadata', {})
            dataset = metadata.get('dataset', dataset)
            threat_model = metadata.get('threat_model', threat_model)
            model_name = metadata.get('model_name', model_name)
            attack_name = metadata.get('attack_name', attack_name)
            attack_lib = metadata.get('attack_lib', attack_lib)
            n_samples = metadata.get('n_samples', len(data.get('adv_success', [])))
        except:
            print("Error: Could not extract metadata from file")
            return False
    
    if not all([dataset, threat_model, model_name, attack_name, attack_lib]):
        print("Error: Missing required metadata (dataset, threat_model, model_name, attack_name, attack_lib)")
        return False
    
    # Create artifact name following convention
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}"
    
    print(f"Uploading to W&B: {artifact_name}")
    print(f"   File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size / 1024:.1f} KB")
    
    try:
        # Check if artifact already exists
        if not overwrite:
            api = _get_wandb_api()
            try:
                existing = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:latest")
                print(f"Warning: Artifact already exists. Use overwrite=True to replace it.")
                return False
            except:
                pass  # Artifact doesn't exist, continue
        
        # Upload to W&B
        with wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="upload-distances") as run:
            artifact = wandb.Artifact(
                name=artifact_name,
                type="precompiled_distances",
                description=f"Precompiled distances for {attack_name} ({attack_lib}) on {model_name} ({dataset}, {threat_model})",
                metadata={
                    "dataset": dataset,
                    "model": model_name,
                    "attack": attack_name,
                    "attack_lib": attack_lib,
                    "threat_model": threat_model,
                    "n_samples": n_samples,
                    "file_size": Path(file_path).stat().st_size
                }
            )
            artifact.add_file(str(file_path))
            run.log_artifact(artifact)
        
        print(f"Successfully uploaded: {artifact_name}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False

def download_precompiled_distances(
    dataset: str,
    threat_model: str,
    model_name: str,
    attack_name: str,
    attack_lib: str,
    n_samples: int = None,
    cache_dir: str = "./cache",
    force_download: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Download precompiled distances from W&B.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Carmon2019Unlabeled')
        attack_name: Attack name (e.g., 'deepfool', 'pgd')
        attack_lib: Library implementing the attack (e.g., 'foolbox', 'torchattacks')
        n_samples: Number of samples (optional, will find latest if not specified)
        cache_dir: Local cache directory
        force_download: Force re-download even if cached
        
    Returns:
        Dictionary with precompiled distances, or None if not found
    """
    
    # Try to find artifact
    if n_samples:
        artifact_name = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-{n_samples}"
        return _download_artifact(artifact_name, dataset, cache_dir, force_download)
    else:
        # Search for any matching artifact (will get latest)
        prefix = f"{dataset}-{threat_model}-{model_name}-{attack_name}-{attack_lib}-"
        return _search_and_download(prefix, dataset, cache_dir, force_download)


def _download_artifact(
    artifact_name: str, 
    dataset: str, 
    cache_dir: str, 
    force_download: bool
) -> Optional[Dict[str, Any]]:
    """Helper to download a specific artifact."""
    file_name = f"{artifact_name}.json"
    
    cache_path = Path(cache_dir) / dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file = cache_path / file_name
    
    # 1. Check local cache first (unless forced)
    if not force_download and local_file.exists():
        try:
            print(f"Loading from cache: {local_file}")
            with open(local_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted cache file, re-downloading...")
    
    # 2. Download from W&B
    print(f"Downloading from W&B: {artifact_name}")
    
    try:
        api = _get_wandb_api()
        artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{artifact_name}:latest")
        
        print(f"   Found artifact: {artifact.name} (v{artifact.version})")
        print(f"   Size: {artifact.size / 1024:.1f} KB")
        print(f"   Samples: {artifact.metadata.get('n_samples', 'Unknown')}")
        
        # Download to cache
        download_path = artifact.download(root=str(cache_path))
        
        # Find the JSON file
        downloaded_file = Path(download_path) / file_name
        if not downloaded_file.exists():
            # Fallback: look for any JSON file
            json_files = list(Path(download_path).glob("*.json"))
            downloaded_file = json_files[0] if json_files else None
        
        if not downloaded_file or not downloaded_file.exists():
            print(f"Error: No JSON file found in downloaded artifact")
            return None
        
        # Load and return data
        with open(downloaded_file, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully downloaded")
        return data
        
    except wandb.errors.CommError:
        print(f"Error: Artifact not found: {artifact_name}")
        return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def _search_and_download(
    prefix: str,
    dataset: str,
    cache_dir: str,
    force_download: bool
) -> Optional[Dict[str, Any]]:
    """Helper to search and download matching artifacts."""
    try:
        api = _get_wandb_api()
        artifact_type = api.artifact_type("precompiled_distances", f"{WANDB_ENTITY}/{WANDB_PROJECT}")
        artifacts = artifact_type.artifacts(per_page=50)
        
        # Find matching artifacts
        matches = [a for a in artifacts if a.name.startswith(prefix)]
        if not matches:
            print(f"Error: No artifacts found matching: {prefix}*")
            return None
        
        # Get latest
        latest = matches[0]
        print(f"Found {len(matches)} matching artifacts, using latest: {latest.name}")
        return _download_artifact(latest.name, dataset, cache_dir, force_download)
        
    except Exception as e:
        print(f"Search failed: {e}")
        return None


# =============================================================================
# OPTIMAL DISTANCES (Lower Envelope) Functions
# =============================================================================

def upload_optimal_distances(
    file_path: Union[str, Path] = None,
    optimal_data: Dict[str, Any] = None,
    dataset: str = None,
    threat_model: str = None,
    model_name: str = None,
    n_samples: int = None,
    overwrite: bool = False
) -> bool:
    """
    Upload optimal distances (lower envelope) to W&B.
    
    Two usage modes:
    1. Pass file_path directly (file already exists)
    2. Pass optimal_data + metadata (automatic file creation)
    
    Args:
        file_path: Path to JSON file (optional if optimal_data provided)
        optimal_data: Dictionary with optimal distances (optional if file_path provided)
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        n_samples: Number of samples (will be inferred from data if not provided)
        overwrite: Whether to overwrite existing artifacts
        
    Returns:
        True if upload successful, False otherwise
        
    Example:
        >>> # Upload from file
        >>> upload_optimal_distances('./optimal/cifar10-linf-Standard-1000.json')
        
        >>> # Upload from data
        >>> upload_optimal_distances(
        ...     optimal_data={'distances': {'linf': [0.1, 0.2, ...]}, ...},
        ...     dataset='cifar10',
        ...     threat_model='linf', 
        ...     model_name='Standard'
        ... )
    """
    
    # Mode 2: Create file from optimal_data
    if optimal_data is not None:
        if not all([dataset, threat_model, model_name]):
            print("Error: Must provide dataset, threat_model, model_name with optimal_data")
            return False
        
        # Infer n_samples from data
        if n_samples is None:
            distances = optimal_data.get('distances', {}).get(threat_model, [])
            n_samples = len(distances) if distances else 0
        
        if n_samples == 0:
            print("Error: Could not determine n_samples from optimal_data")
            return False
        
        # Create temp file
        output_dir = Path('./temp_upload/optimal')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifact_name = f"{dataset}-{threat_model}-{model_name}-{n_samples}"
        file_path = output_dir / f"{artifact_name}.json"
        
        # Add metadata to data
        save_data = {
            **optimal_data,
            'metadata': {
                'dataset': dataset,
                'threat_model': threat_model,
                'model_name': model_name,
                'n_samples': n_samples,
                'type': 'optimal_distances'
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
    else:
        # Mode 1: Use provided file_path
        if file_path is None:
            print("Error: Must provide either file_path or optimal_data")
            return False
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return False
        
        # Extract metadata from file or filename
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            metadata = data.get('metadata', {})
            dataset = metadata.get('dataset') or dataset
            threat_model = metadata.get('threat_model') or threat_model
            model_name = metadata.get('model_name') or model_name
            n_samples = metadata.get('n_samples') or n_samples
            
            # Fallback: parse from filename
            if not all([dataset, threat_model, model_name, n_samples]):
                parts = file_path.stem.split('-')
                if len(parts) >= 4:
                    dataset = dataset or parts[0]
                    threat_model = threat_model or parts[1]
                    model_name = model_name or '-'.join(parts[2:-1])
                    n_samples = n_samples or int(parts[-1])
        except Exception as e:
            print(f"Error: Could not extract metadata from file: {e}")
            return False
    
    if not all([dataset, threat_model, model_name, n_samples]):
        print("Error: Missing required metadata (dataset, threat_model, model_name, n_samples)")
        return False
    
    # Create artifact name
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{n_samples}"
    
    print(f"Uploading optimal distances to W&B: {artifact_name}")
    print(f"   File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size / 1024:.1f} KB")
    
    try:
        # Check if artifact already exists
        if not overwrite:
            api = _get_wandb_api()
            try:
                existing = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT_OPTIMAL}/{artifact_name}:latest")
                print(f"Warning: Artifact already exists. Use overwrite=True to replace it.")
                return False
            except:
                pass  # Artifact doesn't exist, continue
        
        # Upload to W&B
        with wandb.init(project=WANDB_PROJECT_OPTIMAL, entity=WANDB_ENTITY, job_type="upload-optimal") as run:
            artifact = wandb.Artifact(
                name=artifact_name,
                type="optimal_distances",
                description=f"Optimal distances (lower envelope) for {model_name} ({dataset}, {threat_model})",
                metadata={
                    "dataset": dataset,
                    "model": model_name,
                    "threat_model": threat_model,
                    "n_samples": n_samples,
                    "type": "optimal_distances",
                    "file_size": Path(file_path).stat().st_size
                }
            )
            artifact.add_file(str(file_path))
            run.log_artifact(artifact)
        
        print(f"Successfully uploaded optimal distances: {artifact_name}")
        return True
        
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def download_optimal_distances(
    dataset: str,
    threat_model: str,
    model_name: str,
    n_samples: int,
    cache_dir: str = "./cache",
    force_download: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Download optimal distances (lower envelope) from W&B.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        n_samples: Number of samples
        cache_dir: Local cache directory
        force_download: Force re-download even if cached
        
    Returns:
        Dictionary with optimal distances, or None if not found
        
    Example:
        >>> optimal = download_optimal_distances('cifar10', 'linf', 'Standard', 1000)
        >>> distances = optimal['distances']['linf']
    """
    artifact_name = f"{dataset}-{threat_model}-{model_name}-{n_samples}"
    file_name = f"{artifact_name}.json"
    
    # Use separate cache folder for optimal distances
    cache_path = Path(cache_dir) / "optimal" / dataset
    cache_path.mkdir(parents=True, exist_ok=True)
    local_file = cache_path / file_name
    
    # 1. Check local cache first (unless forced)
    if not force_download and local_file.exists():
        try:
            print(f"Loading optimal distances from cache: {local_file}")
            with open(local_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Corrupted cache file, re-downloading...")
    
    # 2. Download from W&B
    print(f"Downloading optimal distances from W&B: {artifact_name}")
    
    try:
        api = _get_wandb_api()
        artifact = api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT_OPTIMAL}/{artifact_name}:latest")
        
        print(f"   Found artifact: {artifact.name} (v{artifact.version})")
        print(f"   Size: {artifact.size / 1024:.1f} KB")
        
        # Download to cache
        download_path = artifact.download(root=str(cache_path))
        
        # Find the JSON file
        downloaded_file = Path(download_path) / file_name
        if not downloaded_file.exists():
            # Fallback: look for any JSON file
            json_files = list(Path(download_path).glob("*.json"))
            downloaded_file = json_files[0] if json_files else None
        
        if not downloaded_file or not downloaded_file.exists():
            print(f"Error: No JSON file found in downloaded artifact")
            return None
        
        # Load and return data
        with open(downloaded_file, 'r') as f:
            data = json.load(f)
        
        print(f"Successfully downloaded optimal distances")
        return data
        
    except wandb.errors.CommError:
        print(f"Error: Optimal distances artifact not found: {artifact_name}")
        return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def upload_directory(
    directory: Union[str, Path],
    dataset: str = None,
    overwrite: bool = False,
    artifact_name: str = None  # For compatibility with test
) -> Dict[str, bool]:
    """
    Upload all JSON files in a directory to W&B.
    Auto-detects metadata from filename: dataset-threat-model-batch.json
    
    Args:
        directory: Directory containing JSON files
        dataset: Filter to specific dataset (None for all)
        overwrite: Whether to overwrite existing artifacts
        artifact_name: Custom artifact name (for test compatibility, usually None)
        
    Returns:
        Dictionary mapping filename to upload success status
        
    Example:
        >>> results = upload_directory('./compiled', dataset='cifar10')
        >>> successful = sum(results.values())
        >>> print(f"Uploaded {successful}/{len(results)} files")
    """
    
    directory = Path(directory)
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return {}
    
    results = {}
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {directory}")
        return {}
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    # Handle test case with custom artifact_name
    if artifact_name:
        print(f"Note: Custom artifact_name '{artifact_name}' provided (test mode)")
    
    for file_path in json_files:
        # Skip files that don't match the expected pattern
        if file_path.name.count("-") < 3:
            print(f"Skipping {file_path.name} (invalid format)")
            continue
        
        try:
            # Parse metadata from filename: dataset-threat-model-batch.json
            parts = file_path.stem.split("-")
            file_dataset = parts[0]
            file_threat_model = parts[1]
            file_batch_size = int(parts[-1])
            file_model_name = "-".join(parts[2:-1])
            
            # Filter by dataset if specified
            if dataset and file_dataset != dataset:
                print(f"Skipping {file_path.name} (dataset mismatch)")
                continue
            
            print(f"\nProcessing: {file_path.name}")
            success = upload_precompiled_distances(
                file_path, file_dataset, file_threat_model, 
                file_model_name, file_batch_size, overwrite
            )
            results[file_path.name] = success
            
        except (ValueError, IndexError) as e:
            print(f"Error: Invalid filename format: {file_path.name}")
            results[file_path.name] = False
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    print(f"\nSUMMARY: {successful}/{total} files uploaded successfully")
    
    if successful < total:
        failed = [name for name, success in results.items() if not success]
        print(f"Failed uploads: {failed}")
    
    return results

def list_available_distances(dataset: str = None) -> List[Dict[str, Any]]:
    """
    List all available precompiled distances on W&B.
    
    Args:
        dataset: Filter to specific dataset (None for all)
        
    Returns:
        List of dictionaries with artifact metadata
        
    Example:
        >>> artifacts = list_available_distances('cifar10')
        >>> for art in artifacts:
        ...     print(f"{art['model']} - {art['threat_model']}")
    """
    
    try:
        # Since direct API listing is problematic, use a workaround
        # Check what artifacts are actually available by trying known patterns
        from .wandb_utils import get_precompiled_distances
        
        known_datasets = ['cifar10', 'imagenet']
        known_threats = ['l0', 'l1', 'l2', 'linf']  
        known_models = ['standard', 'wong_2020', 'salman_2020', 'debenedetti_2022']
        
        artifacts_found = []
        
        for test_dataset in known_datasets:
            if dataset and test_dataset != dataset:
                continue
                
            for threat in known_threats:
                for model in known_models:
                    try:
                        # Test if this combination exists by trying to download
                        test_data = get_precompiled_distances(
                            dataset=test_dataset,
                            threat_model=threat,
                            model_name=model,
                            batch_size=1000,
                            cache_dir="/tmp/wandb_test"
                        )
                        
                        if test_data is not None:
                            artifacts_found.append({
                                'name': f"{test_dataset}-{threat}-{model}-1000",
                                'dataset': test_dataset,
                                'model': model,
                                'threat_model': threat,
                                'batch_size': 1000,
                                'num_samples': len(test_data),
                                'size_kb': 0,  # Unknown
                                'version': 'v0',  # Unknown
                                'created_at': 'unknown'
                            })
                    except:
                        continue
        
        return artifacts_found
        
        results = []
        for artifact in artifacts:
            try:
                # Debug info
                print(f"Processing artifact: {getattr(artifact, 'name', 'UNKNOWN')}")
                
                # Safely handle metadata
                if hasattr(artifact, 'metadata'):
                    metadata = artifact.metadata or {}
                else:
                    print(f"Artifact has no metadata attribute")
                    metadata = {}
                
                # Filter by dataset if specified
                if dataset and metadata.get('dataset') != dataset:
                    continue
                
                results.append({
                    'name': getattr(artifact, 'name', 'unknown'),
                    'dataset': metadata.get('dataset', 'unknown'),
                    'model': metadata.get('model', 'unknown'),
                    'threat_model': metadata.get('threat_model', 'unknown'),
                    'batch_size': metadata.get('batch_size', 'unknown'),
                    'num_samples': metadata.get('num_samples', 'unknown'),
                    'size_kb': getattr(artifact, 'size', 0) / 1024 if getattr(artifact, 'size', 0) else 0,
                    'version': getattr(artifact, 'version', 'unknown'),
                    'created_at': getattr(artifact, 'created_at', 'unknown')
                })
            except Exception as e:
                print(f"Error processing artifact: {e}")
                continue
        
        return sorted(results, key=lambda x: (x['dataset'], x['threat_model'], x['model']))
        
    except Exception as e:
        print(f"Failed to list artifacts: {e}")
        return []

# Mantieni compatibilità con il codice esistente
get_precompiled_distances = download_precompiled_distances

# CLI se eseguito direttamente
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AttackBench W&B Manager")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload distances')
    upload_parser.add_argument('file', help='JSON file to upload')
    upload_parser.add_argument('--dataset', required=True)
    upload_parser.add_argument('--threat-model', required=True) 
    upload_parser.add_argument('--model', required=True)
    upload_parser.add_argument('--batch-size', type=int, required=True)
    upload_parser.add_argument('--overwrite', action='store_true')
    
    # Upload directory command
    upload_dir_parser = subparsers.add_parser('upload-dir', help='Upload directory')
    upload_dir_parser.add_argument('directory', help='Directory containing JSON files')
    upload_dir_parser.add_argument('--dataset', help='Filter by dataset')
    upload_dir_parser.add_argument('--overwrite', action='store_true')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download distances')
    download_parser.add_argument('--dataset', required=True)
    download_parser.add_argument('--threat-model', required=True)
    download_parser.add_argument('--model', required=True) 
    download_parser.add_argument('--batch-size', type=int, required=True)
    download_parser.add_argument('--cache-dir', default='./cache')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available distances')
    list_parser.add_argument('--dataset', help='Filter by dataset')
    
    args = parser.parse_args()
    
    if args.command == 'upload':
        upload_precompiled_distances(
            args.file, args.dataset, args.threat_model, 
            args.model, args.batch_size, args.overwrite
        )
    elif args.command == 'upload-dir':
        upload_directory(args.directory, args.dataset, args.overwrite)
    elif args.command == 'download':
        data = download_precompiled_distances(
            args.dataset, args.threat_model, args.model, 
            args.batch_size, args.cache_dir
        )
        if data:
            print(f"Downloaded {len(data)} distances")
    elif args.command == 'list':
        artifacts = list_available_distances(args.dataset)
        for art in artifacts:
            print(f"{art['dataset']}-{art['threat_model']}-{art['model']}-{art['batch_size']} (v{art['version']})")