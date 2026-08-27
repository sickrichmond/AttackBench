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
