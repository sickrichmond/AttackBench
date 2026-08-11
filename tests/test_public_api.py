"""The package must be usable with only its core dependencies installed."""
import pytest


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
        assert 'pip install attackbenchlib[attacks]' in str(e)


def test_unknown_attribute_still_raises_attribute_error():
    import attackbench

    with pytest.raises(AttributeError):
        attackbench.definitely_not_a_thing


def test_registry_lists_attacks_when_the_libraries_are_installed():
    registry = pytest.importorskip(
        'attackbench.attacks.registry',
        reason='attack libraries not installed (pip install attackbenchlib[attacks])')

    attacks = registry.list_attacks()
    assert attacks, 'no attacks discovered'

    for norm in ('l0', 'l1', 'l2', 'linf'):
        for lib, name in registry.list_attacks(threat_model=norm):
            # every listed attack must be buildable for the norm it is listed under
            registry.get_attack(lib=lib, attack=name, threat_model=norm)
