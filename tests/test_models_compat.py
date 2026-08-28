"""Compatibility tests that do not require the optional models extra."""

from types import SimpleNamespace

import pytest

from attackbench.models import robustbench_compat


def test_robustbench_loader_aliases_pyautoattack(monkeypatch):
    marker = object()
    modules = {
        "pyautoattack": SimpleNamespace(__name__="pyautoattack"),
        "pyautoattack.state": SimpleNamespace(__name__="pyautoattack.state"),
        "robustbench": SimpleNamespace(load_model=marker),
    }

    def fake_import(name):
        if name == "autoattack":
            raise ModuleNotFoundError("No module named 'autoattack'", name=name)
        return modules[name]

    monkeypatch.delitem(robustbench_compat.sys.modules, "autoattack", raising=False)
    monkeypatch.delitem(
        robustbench_compat.sys.modules, "autoattack.state", raising=False
    )
    monkeypatch.setattr(robustbench_compat.importlib, "import_module", fake_import)

    try:
        assert robustbench_compat.get_robustbench_loader() is marker
        assert robustbench_compat.sys.modules["autoattack"] is modules["pyautoattack"]
        assert (
            robustbench_compat.sys.modules["autoattack.state"]
            is modules["pyautoattack.state"]
        )
    finally:
        robustbench_compat.sys.modules.pop("autoattack", None)
        robustbench_compat.sys.modules.pop("autoattack.state", None)


def test_robustbench_loader_does_not_hide_nested_import_errors(monkeypatch):
    def broken_import(name):
        raise ModuleNotFoundError("No module named 'dependency'", name="dependency")

    monkeypatch.setattr(robustbench_compat.importlib, "import_module", broken_import)

    with pytest.raises(ModuleNotFoundError, match="dependency"):
        robustbench_compat.get_robustbench_loader()


def test_mit_clean_model_registry_restores_external_asset_models():
    from attackbench.models.registry import MODEL_CONFIGS

    assert MODEL_CONFIGS["stutz_2020"] == {
        "name": "Stutz2020CCAT",
        "source": "original",
        "dataset": "cifar10",
        "threat_model": "Linf",
    }
    assert MODEL_CONFIGS["xiao_2020"] == {
        "name": "Xiao2020KWTA",
        "source": "original",
        "dataset": "cifar10",
        "threat_model": "Linf",
    }
    assert MODEL_CONFIGS["wang_2023_small"] == {
        "name": "Wang2023Better_WRN-28-10",
        "source": "robustbench",
        "dataset": "cifar10",
        "threat_model": "Linf",
    }
    assert MODEL_CONFIGS["wang_2023_large"] == {
        "name": "Wang2023Better_WRN-70-16",
        "source": "robustbench",
        "dataset": "cifar10",
        "threat_model": "Linf",
    }


def test_external_model_architectures_forward_and_kwta_sparsity():
    import torch

    from attackbench.models.original.compat_architectures import (
        KWTA2d,
        Stutz2020CCAT,
        Xiao2020KWTA,
    )

    inputs = torch.rand(1, 3, 32, 32)
    assert Stutz2020CCAT()(inputs).shape == (1, 10)
    assert Xiao2020KWTA()(inputs).shape == (1, 10)

    activations = torch.arange(20.0).reshape(1, 2, 2, 5)
    sparse = KWTA2d(0.1)(activations)
    assert torch.count_nonzero(sparse).item() == 2
    assert sparse.max().item() == 19


def test_stutz_checkpoint_requires_explicit_license_acceptance(tmp_path, monkeypatch):
    from attackbench.models.original.external import load_stutz2020

    monkeypatch.delenv("ATTACKBENCH_ACCEPT_STUTZ2020_LICENSE", raising=False)
    with pytest.raises(PermissionError, match="noncommercial"):
        load_stutz2020("Stutz2020CCAT", checkpoint_path=tmp_path / "checkpoint.pth")


def test_xiao_checkpoint_must_be_supplied(monkeypatch):
    from attackbench.models.original.external import load_xiao2020

    monkeypatch.delenv("ATTACKBENCH_XIAO2020_CHECKPOINT", raising=False)
    with pytest.raises(FileNotFoundError, match="does not redistribute"):
        load_xiao2020("Xiao2020KWTA")


def test_external_loader_uses_restricted_deserialization(tmp_path, monkeypatch):
    import torch
    from torch import nn

    from attackbench.models.original import external

    checkpoint = tmp_path / "xiao.pth"
    checkpoint.touch()
    network = nn.Linear(1, 1)
    calls = {}

    def restricted_load(path, **kwargs):
        calls["path"] = path
        calls.update(kwargs)
        return network.state_dict()

    monkeypatch.setattr(external, "_verify", lambda path, expected: None)
    monkeypatch.setattr(external, "Xiao2020KWTA", lambda fraction: network)
    monkeypatch.setattr(external.torch, "load", restricted_load)

    loaded = external.load_xiao2020("Xiao2020KWTA", checkpoint_path=checkpoint)

    assert loaded is network
    assert calls == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": True,
    }
    assert isinstance(loaded, torch.nn.Module)
