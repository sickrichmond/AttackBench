"""Load optional checkpoints that AttackBench cannot redistribute under MIT."""

import hashlib
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Union

import torch
from torch import nn

from .compat_architectures import Stutz2020CCAT, Xiao2020KWTA

PathLike = Union[str, os.PathLike]

_STUTZ_URL = "https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/cifar10_ccat.zip"
_STUTZ_LICENSE_URL = (
    "https://github.com/davidstutz/confidence-calibrated-adversarial-training"
    "#license"
)
_STUTZ_ARCHIVE_SHA256 = (
    "68d88e03a2a924b83f28c105904653f8a7bc510e2a6e48526e161584d5e96298"
)
_STUTZ_CHECKPOINT_SHA256 = (
    "ca857a0f563d60a9762b8cf08e8efc5c9ebbae558166a1d8a7b1c0efa1d48611"
)
_XIAO_CHECKPOINT_URL = (
    "https://github.com/wielandbrendel/robustness_workshop/releases/download/"
    "v0.0.1/kwta_spresnet18_0.1_cifar_adv.pth"
)
_XIAO_CHECKPOINT_SHA256 = (
    "112d953be871c3991117f8ba1599e978623352f70800c3a4d91aa434039c72de"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checkpoint checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _validate_cifar10(dataset: str) -> None:
    if dataset.lower() != "cifar10":
        raise ValueError("This checkpoint is available only for CIFAR-10")


def _download_stutz_checkpoint() -> Path:
    cache_dir = Path(torch.hub.get_dir()) / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "stutz2020_ccat_classifier.pth.tar"
    if destination.exists():
        _verify(destination, _STUTZ_CHECKPOINT_SHA256)
        return destination

    with tempfile.TemporaryDirectory(dir=str(cache_dir)) as temporary:
        archive_path = Path(temporary) / "cifar10_ccat.zip"
        checkpoint_path = Path(temporary) / "classifier.pth.tar"
        print(f"Downloading {_STUTZ_URL}...")
        urllib.request.urlretrieve(_STUTZ_URL, archive_path)
        _verify(archive_path, _STUTZ_ARCHIVE_SHA256)
        with zipfile.ZipFile(archive_path) as archive:
            # Read a single fixed member: never extract arbitrary archive paths.
            checkpoint_path.write_bytes(archive.read("classifier.pth.tar"))
        _verify(checkpoint_path, _STUTZ_CHECKPOINT_SHA256)
        os.replace(str(checkpoint_path), str(destination))
    return destination


def load_stutz2020(
    model: str,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    checkpoint_path: Optional[PathLike] = None,
    accept_license: bool = False,
) -> nn.Module:
    """Load Stutz2020CCAT after explicit acceptance of its asset license."""
    del threat_model
    _validate_cifar10(dataset)
    if model != "Stutz2020CCAT":
        raise ValueError(f"Unsupported Stutz model: {model}")
    if not (accept_license or _env_truthy("ATTACKBENCH_ACCEPT_STUTZ2020_LICENSE")):
        raise PermissionError(
            "The Stutz2020 checkpoint is restricted to noncommercial research, "
            "education, and artistic projects. Review its terms at "
            f"{_STUTZ_LICENSE_URL}, then pass accept_license=True or set "
            "ATTACKBENCH_ACCEPT_STUTZ2020_LICENSE=1 if your use complies."
        )

    configured_path = checkpoint_path or os.environ.get(
        "ATTACKBENCH_STUTZ2020_CHECKPOINT"
    )
    path = Path(configured_path).expanduser() if configured_path else None
    if path is None:
        path = _download_stutz_checkpoint()
    if not path.is_file():
        raise FileNotFoundError(f"Stutz2020 checkpoint not found: {path}")
    _verify(path, _STUTZ_CHECKPOINT_SHA256)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or not isinstance(
        checkpoint.get("model"), Mapping
    ):
        raise ValueError("Unexpected Stutz2020 checkpoint format")
    network = Stutz2020CCAT()
    network.load_state_dict(checkpoint["model"], strict=True)
    return network


def load_xiao2020(
    model: str,
    dataset: str = "cifar10",
    threat_model: str = "Linf",
    checkpoint_path: Optional[PathLike] = None,
    accept_license: bool = False,
) -> nn.Module:
    """Load Xiao2020KWTA from an explicitly supplied external checkpoint."""
    del threat_model, accept_license
    _validate_cifar10(dataset)
    if model != "Xiao2020KWTA":
        raise ValueError(f"Unsupported Xiao model: {model}")
    configured_path = checkpoint_path or os.environ.get(
        "ATTACKBENCH_XIAO2020_CHECKPOINT"
    )
    if not configured_path:
        raise FileNotFoundError(
            "Xiao2020's upstream repository does not state a software/checkpoint "
            "license, so AttackBench does not redistribute or automatically download "
            "the checkpoint. Obtain permission if needed, download it yourself from "
            f"{_XIAO_CHECKPOINT_URL}, and pass checkpoint_path=... or set "
            "ATTACKBENCH_XIAO2020_CHECKPOINT."
        )
    path = Path(configured_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Xiao2020 checkpoint not found: {path}")
    _verify(path, _XIAO_CHECKPOINT_SHA256)

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Unexpected Xiao2020 checkpoint format")
    network = Xiao2020KWTA(fraction=0.1)
    network.load_state_dict(state_dict, strict=True)
    return network
