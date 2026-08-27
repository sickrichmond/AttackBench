"""The package must be usable with only its core dependencies installed."""

import importlib
import os

import pytest


def _registry_or_skip():
    try:
        return importlib.import_module("attackbench.attacks.registry")
    except ImportError as exc:
        if os.environ.get("ATTACKBENCH_REQUIRE_ATTACKS") == "1":
            pytest.fail(
                f"attack libraries were required by this job but the registry could "
                f"not be imported: {exc!r}"
            )
        pytest.skip(
            "attack libraries not installed (pip install attackbenchlib[attacks])"
        )


def test_core_api_imports_without_the_optional_extras():
    import attackbench

    assert attackbench.__version__
    assert callable(attackbench.run_attack)
    assert callable(attackbench.get_loader)
    assert callable(attackbench.create_custom_attack)
    assert attackbench.DEFAULT_QUERY_BUDGET == 2000


def test_missing_extra_gives_an_actionable_error():
    import attackbench

    try:
        attackbench.get_attack
    except ImportError as e:
        assert "pip install attackbenchlib[attacks]" in str(e)


def test_unknown_attribute_still_raises_attribute_error():
    import attackbench

    with pytest.raises(AttributeError):
        attackbench.definitely_not_a_thing


def test_registry_lists_attacks_when_the_libraries_are_installed():
    registry = _registry_or_skip()

    attacks = registry.list_attacks()
    assert attacks, "no attacks discovered"

    for norm in ("l0", "l1", "l2", "linf"):
        for lib, name in registry.list_attacks(threat_model=norm):
            # every listed attack must be buildable for the norm it is listed under
            registry.get_attack(lib=lib, attack=name, threat_model=norm)


def test_registry_uses_norm_specific_original_configs():
    registry = _registry_or_skip()

    def configured_attack(name, norm):
        wrapped = registry.get_attack(lib="original", attack=name, threat_model=norm)
        return wrapped.keywords["attack"]

    apgd_l1 = configured_attack("apgd", "l1")
    assert apgd_l1.keywords["threat_model"] == "l1"
    assert apgd_l1.keywords["eps"] == 10
    assert apgd_l1.keywords["use_largereps"] is True

    apgd_minimal_l1 = configured_attack("apgd_minimal", "l1")
    assert apgd_minimal_l1.keywords["attack"].keywords["use_largereps"] is True

    fmn_linf = configured_attack("fmn", "linf")
    assert fmn_linf.keywords["threat_model"] == "linf"
    assert fmn_linf.keywords["max_stepsize"] == 10


def test_registry_does_not_advertise_undeclared_norms():
    registry = _registry_or_skip()

    by_norm = {
        norm: set(registry.list_attacks(threat_model=norm, lib="original"))
        for norm in ("l0", "l1", "l2", "linf")
    }
    for attack in ("pgd0_minimal", "sigma_zero"):
        assert ("original", attack) in by_norm["l0"]
        assert all(
            ("original", attack) not in by_norm[norm] for norm in ("l1", "l2", "linf")
        )

    assert ("original", "superdeepfool") in by_norm["l2"]
    with pytest.raises(ValueError, match="not available"):
        registry.get_attack("original", "pgd0_minimal", "l2")
