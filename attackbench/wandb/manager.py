"""
AttackBench W&B Manager - User-friendly API for precompiled distances.

Provides simple upload/download functions for managing precompiled attack distances.
"""
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union

import wandb

# Configuration constants
WANDB_ENTITY = "attackbench"
WANDB_PROJECT = "attackbench-precompiled-distancies"
WANDB_PROJECT_OPTIMAL = "attackbench-optimal-distancies"
PROTOCOL_VERSION = 2
DISTANCE_SEMANTICS = "best_observed"
PRECOMPILED_RESULT_FIELDS = (
    "distances",
    "final_distances",
    "adv_success",
    "ori_success",
    "correct",
    "hashes",
    "original_predictions",
    "adversarial_predictions",
    "num_forwards",
    "num_backwards",
    "times",
    "box_failures",
    "batch_failures",
    "targeted",
    "query_budget",
)


def _has_wandb_credentials() -> bool:
    """Check if W&B credentials are available (without prompting)."""
    # Check environment variables
    if os.environ.get("WANDB_API_KEY") or os.environ.get("ATTACKBENCH_WANDB_KEY"):
        return True
    # Check netrc file (where wandb login stores credentials)
    netrc_path = Path.home() / ".netrc"
    if netrc_path.exists():
        try:
            content = netrc_path.read_text()
            if "api.wandb.ai" in content:
                return True
        except:
            pass
    return False


def _get_wandb_api(require_auth: bool = False) -> Optional[wandb.Api]:
    """
    Return a W&B API client, or None if no credentials are available.

    Args:
        require_auth: If True, raises error when no credentials found (for uploads).
                      If False, returns None when no credentials found (for downloads).

    For uploads, set WANDB_API_KEY environment variable.
    For downloads, credentials are optional but required to access W&B artifacts.
    """
    if not _has_wandb_credentials():
        if require_auth:
            raise ValueError(
                "W&B authentication required for this operation. "
                "Set WANDB_API_KEY or ATTACKBENCH_WANDB_KEY environment variable, "
                "or run 'wandb login' to authenticate."
            )
        return None
    
    # Credentials available, create API client
    api_key = os.environ.get("WANDB_API_KEY") or os.environ.get("ATTACKBENCH_WANDB_KEY")
    if api_key:
        return wandb.Api(api_key=api_key)
    return wandb.Api()


def _make_artifact_name(*parts) -> str:
    """Build a W&B artifact name from parts, normalised to lowercase."""
    return "-".join(str(p).lower() for p in parts)


def _validate_precompiled_data(
    data: Optional[Dict[str, Any]], artifact_name: str
) -> Optional[Dict[str, Any]]:
    """Reject legacy artifacts that cannot reproduce a 2.x run result."""
    if data is None:
        return None
    missing = [field for field in PRECOMPILED_RESULT_FIELDS if field not in data]
    if missing:
        warnings.warn(
            f"Ignoring incompatible pre-2.0 precompiled artifact '{artifact_name}' "
            f"(missing fields: {', '.join(missing)}). Re-run and upload it with "
            "AttackBench 2.x.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return data


def _validate_optimal_data(
    data: Optional[Dict[str, Any]], artifact_name: str
) -> Optional[Dict[str, Any]]:
    """Reject lower envelopes built with pre-2.0 distance semantics."""
    if data is None:
        return None
    metadata = data.get("metadata", {})
    compatible = (
        metadata.get("protocol_version") == PROTOCOL_VERSION
        and metadata.get("distance_semantics") == DISTANCE_SEMANTICS
        and metadata.get("format") == "hash_based"
        and isinstance(data.get("distances"), dict)
    )
    if not compatible:
        warnings.warn(
            f"Ignoring incompatible pre-2.0 optimal-distance artifact '{artifact_name}'. "
            "Rebuild the lower envelope from AttackBench 2.x attack results.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return data


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
    
    Three usage modes:
    1. Pass file_path directly (legacy)
    2. Pass attack_data only (metadata extracted automatically from attack_data['metadata'])
    3. Pass attack_data + explicit metadata (overrides extracted values)
    
    Args:
        file_path: Path to JSON file (optional if attack_data provided)
        attack_data: Raw attack results dict from run_attack() (optional if file_path provided)
        dataset: Dataset name (optional, extracted from attack_data if not provided)
        threat_model: Threat model (optional, extracted from attack_data if not provided)
        model_name: Model name (optional, extracted from attack_data if not provided)
        attack_name: Attack name (optional, extracted from attack_data if not provided)
        attack_lib: Library implementing the attack (optional, extracted from attack_data if not provided)
        overwrite: Whether to overwrite existing artifacts
        
    Returns:
        True if upload successful, False otherwise
    """
    
    # Mode 2/3: Create file from attack_data
    if attack_data is not None:
        if _validate_precompiled_data(attack_data, "attack_data") is None:
            return False
        # Auto-extract metadata from attack_data if not explicitly provided
        meta = attack_data.get('metadata', {})
        dataset = dataset or meta.get('dataset')
        threat_model = threat_model or meta.get('threat_model')
        model_name = model_name or meta.get('model_name')
        attack_name = attack_name or meta.get('attack_name')
        attack_lib = attack_lib or meta.get('attack_lib')
        
        if not all([dataset, threat_model, model_name, attack_name, attack_lib]):
            print("Error: Missing metadata. Provide dataset, threat_model, model_name, attack_name, attack_lib "
                  "either explicitly or via attack_data['metadata'] (from run_attack)")
            return False
        
        from ..metrics.storage import save_precompiled_distances
        file_path, metadata = save_precompiled_distances(
            attack_data, dataset=dataset, threat_model=threat_model, model_name=model_name,
            attack_name=attack_name, attack_lib=attack_lib
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
            if _validate_precompiled_data(data, str(file_path)) is None:
                return False
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
    
    # Verify authentication before attempting upload
    _get_wandb_api(require_auth=True)
    
    # Create artifact name following convention (all lowercase)
    artifact_name = _make_artifact_name(dataset, threat_model, model_name, attack_name, attack_lib, n_samples)
    
    print(f"Uploading to W&B: {artifact_name}")
    print(f"   File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size / 1024:.1f} KB")
    
    try:
        # Check if artifact already exists
        if not overwrite:
            api = _get_wandb_api(require_auth=True)
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
                    "file_size": Path(file_path).stat().st_size,
                    "protocol_version": PROTOCOL_VERSION,
                    "distance_semantics": DISTANCE_SEMANTICS,
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
        artifact_name = _make_artifact_name(dataset, threat_model, model_name, attack_name, attack_lib, n_samples)
        return _download_artifact(artifact_name, dataset, cache_dir, force_download)
    else:
        # Search for any matching artifact (will get latest)
        prefix = _make_artifact_name(dataset, threat_model, model_name, attack_name, attack_lib) + "-"
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
                data = json.load(f)
            validated = _validate_precompiled_data(data, artifact_name)
            if validated is not None:
                return validated
            print("Cached artifact is incompatible; checking W&B for a replacement.")
        except json.JSONDecodeError:
            print(f"Warning: Corrupted cache file, re-downloading...")
    
    # 2. Download from W&B
    print(f"Downloading from W&B: {artifact_name}")
    
    api = _get_wandb_api()
    if api is None:
        print(f"W&B credentials not found. Skipping download.")
        print(f"Run 'wandb login' or set WANDB_API_KEY to enable W&B access.")
        return None
    
    try:
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
        return _validate_precompiled_data(data, artifact_name)
        
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
    api = _get_wandb_api()
    if api is None:
        print(f"W&B credentials not found. Skipping search.")
        return None
    
    try:
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
    overwrite: bool = False
) -> bool:
    """
    Upload optimal distances (lower envelope) to W&B.
    
    Optimal distances are stored as hash-based lookup tables computed on the
    FULL dataset. This allows matching with any subset of samples via their
    per-image SHA-512 hashes.
    
    Data format:
        distances: {threat_model: {image_hash: distance, ...}}
    
    Two usage modes:
    1. Pass file_path directly (file already exists)
    2. Pass optimal_data + metadata (automatic file creation)
    
    Args:
        file_path: Path to JSON file (optional if optimal_data provided)
        optimal_data: Dictionary with optimal distances (optional if file_path provided).
            Must contain 'distances' as {threat_model: {hash: distance}} dict.
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        overwrite: Whether to overwrite existing artifacts
        
    Returns:
        True if upload successful, False otherwise
        
    Example:
        >>> upload_optimal_distances(
        ...     optimal_data={'distances': {'linf': {'abc123...': 0.1, 'def456...': 0.2}}},
        ...     dataset='cifar10', threat_model='linf', model_name='Standard'
        ... )
    """
    
    # Mode 2: Create file from optimal_data
    if optimal_data is not None:
        if not all([dataset, threat_model, model_name]):
            print("Error: Must provide dataset, threat_model, model_name with optimal_data")
            return False
        
        # Infer n_samples from data (hash-based dict)
        distances = optimal_data.get('distances', {}).get(threat_model, {})
        if isinstance(distances, dict):
            n_samples = len(distances)
        elif isinstance(distances, list):
            n_samples = len(distances)
        else:
            n_samples = 0
        
        if n_samples == 0:
            print("Error: Could not determine n_samples from optimal_data")
            return False
        
        # Create temp file
        output_dir = Path('./temp_upload/optimal')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Artifact name no longer includes n_samples (always full dataset)
        artifact_name = _make_artifact_name(dataset, threat_model, model_name)
        file_path = output_dir / f"{artifact_name}.json"
        
        # Add metadata to data
        save_data = {
            **optimal_data,
            'metadata': {
                'dataset': dataset,
                'threat_model': threat_model,
                'model_name': model_name,
                'n_samples': n_samples,
                'type': 'optimal_distances',
                'format': 'hash_based',
                'protocol_version': PROTOCOL_VERSION,
                'distance_semantics': DISTANCE_SEMANTICS,
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
        
        # Extract metadata from file
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if _validate_optimal_data(data, str(file_path)) is None:
                return False
            metadata = data.get('metadata', {})
            dataset = metadata.get('dataset') or dataset
            threat_model = metadata.get('threat_model') or threat_model
            model_name = metadata.get('model_name') or model_name
        except Exception as e:
            print(f"Error: Could not extract metadata from file: {e}")
            return False
    
    if not all([dataset, threat_model, model_name]):
        print("Error: Missing required metadata (dataset, threat_model, model_name)")
        return False
    
    # Verify authentication before attempting upload
    _get_wandb_api(require_auth=True)
    
    # Artifact name without n_samples (always full dataset)
    artifact_name = _make_artifact_name(dataset, threat_model, model_name)
    
    print(f"Uploading optimal distances to W&B: {artifact_name}")
    print(f"   File: {file_path}")
    print(f"   Size: {Path(file_path).stat().st_size / 1024:.1f} KB")
    
    try:
        # Check if artifact already exists
        if not overwrite:
            api = _get_wandb_api(require_auth=True)
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
                description=f"Optimal distances (lower envelope) for {model_name} ({dataset}, {threat_model}) - hash-based, full dataset",
                metadata={
                    "dataset": dataset,
                    "model": model_name,
                    "threat_model": threat_model,
                    "type": "optimal_distances",
                    "format": "hash_based",
                    "protocol_version": PROTOCOL_VERSION,
                    "distance_semantics": DISTANCE_SEMANTICS,
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
    cache_dir: str = "./cache",
    force_download: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Download optimal distances (lower envelope) from W&B.
    
    Optimal distances are stored as hash-based lookup tables:
        distances: {threat_model: {image_hash: distance, ...}}
    
    This allows matching with any subset of samples via per-image SHA-512 hashes.
    
    Args:
        dataset: Dataset name (e.g., 'cifar10')
        threat_model: Threat model (e.g., 'linf', 'l2')
        model_name: Model name (e.g., 'Standard')
        cache_dir: Local cache directory
        force_download: Force re-download even if cached
        
    Returns:
        Dictionary with optimal distances (hash-based), or None if not found
        
    Example:
        >>> optimal = download_optimal_distances('cifar10', 'linf', 'Standard')
        >>> hash_to_dist = optimal['distances']['linf']  # {hash: distance, ...}
    """
    # Artifact name without n_samples (always full dataset)
    artifact_name = _make_artifact_name(dataset, threat_model, model_name)
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
                data = json.load(f)
            validated = _validate_optimal_data(data, artifact_name)
            if validated is not None:
                return validated
            print("Cached envelope is incompatible; checking W&B for a replacement.")
        except json.JSONDecodeError:
            print(f"Warning: Corrupted cache file, re-downloading...")
    
    # 2. Download from W&B
    print(f"Downloading optimal distances from W&B: {artifact_name}")
    
    api = _get_wandb_api()
    if api is None:
        print(f"W&B credentials not found. Cannot download optimal distances.")
        print(f"Run 'wandb login' or set WANDB_API_KEY to enable W&B access.")
        return None
    
    try:
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
        return _validate_optimal_data(data, artifact_name)
        
    except wandb.errors.CommError:
        print(f"Error: Optimal distances artifact not found: {artifact_name}")
        return None
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def update_optimal_distances(
    attack_results: Dict[str, Any],
    dataset: str = None,
    threat_model: str = None,
    model_name: str = None,
    cache_dir: str = "./cache",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Fold one attack run into the optimal distances (lower envelope) stored on W&B.

    This is stage 5 of AttackBench: a new attack only has to update a*, the per-sample
    best known distance — previously benchmarked attacks are never re-run. The envelope
    is keyed by the per-sample SHA-512 hash, so runs on different subsets of the same
    dataset compose without any ordering assumption.

    Args:
        attack_results: output of run_attack() — needs 'distances', 'hashes' and 'metadata'
        dataset / threat_model / model_name: override the values in attack_results['metadata']
        cache_dir: local cache directory for the W&B download
        dry_run: compute and report the update without uploading

    Returns:
        Dict with 'improved' (samples whose best distance went down), 'added' (samples not
        previously in the envelope), 'n_samples' and 'distances' (the merged envelope).
    """
    meta = attack_results.get('metadata', {})
    if (
        meta.get('protocol_version') != PROTOCOL_VERSION
        or meta.get('distance_semantics') != DISTANCE_SEMANTICS
    ):
        raise ValueError(
            "attack_results must come from AttackBench 2.x with best-observed (d*) "
            "distance semantics; legacy results cannot update the current envelope."
        )
    dataset = dataset or meta.get('dataset')
    threat_model = threat_model or meta.get('threat_model')
    model_name = model_name or meta.get('model_name')
    if not all([dataset, threat_model, model_name]):
        raise ValueError(
            "dataset, threat_model and model_name are required (pass them explicitly or "
            "run the attack through run_attack(), which records them in ['metadata'])."
        )

    hashes = attack_results.get('hashes', [])
    new_distances = attack_results.get('distances', {}).get(threat_model, [])
    if not hashes or not new_distances:
        raise ValueError(f"attack_results has no hashes/distances for threat model '{threat_model}'")
    if len(hashes) != len(new_distances):
        raise ValueError(f"hashes ({len(hashes)}) and distances ({len(new_distances)}) disagree")

    current = download_optimal_distances(dataset, threat_model, model_name, cache_dir=cache_dir)
    envelope = dict((current or {}).get('distances', {}).get(threat_model, {}))

    improved, added = 0, 0
    for h, d in zip(hashes, new_distances):
        if d is None or not float(d) == float(d):  # skip NaN
            continue
        previous = envelope.get(h)
        if previous is None:
            envelope[h] = float(d)
            added += 1
        elif float(d) < float(previous):
            envelope[h] = float(d)
            improved += 1

    print(f"[AttackBench] Lower envelope for {dataset}/{threat_model}/{model_name}: "
          f"{improved} improved, {added} new, {len(envelope)} samples total")

    if not dry_run and (improved or added):
        upload_optimal_distances(
            optimal_data={'distances': {threat_model: envelope}},
            dataset=dataset, threat_model=threat_model, model_name=model_name,
            overwrite=True,
        )

    return {'improved': improved, 'added': added,
            'n_samples': len(envelope), 'distances': {threat_model: envelope}}
