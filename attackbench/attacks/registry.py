import inspect
import sys
import warnings
from collections import defaultdict
from functools import lru_cache, partial
from typing import Callable

from .art import configs as art_configs
from .cleverhans import configs as cleverhans_configs
from .foolbox import configs as foolbox_configs
from .original import configs as original_configs

try:
    from .torchattacks import configs as torchattacks_configs

    _has_torchattacks = True
except ImportError:  # pragma: no cover - optional dependency
    torchattacks_configs = None
    _has_torchattacks = False

try:
    from .adv_lib import configs as adv_lib_configs

    _has_adv_lib = True
except ImportError:  # pragma: no cover - optional dependency
    adv_lib_configs = None
    _has_adv_lib = False

try:
    from .deeprobust import configs as deeprobust_configs

    _has_deeprobust = True
except ImportError:  # pragma: no cover - optional dependency
    deeprobust_configs = None
    _has_deeprobust = False

library_modules = {
    "art": art_configs,
    "cleverhans": cleverhans_configs,
    "foolbox": foolbox_configs,
    "original": original_configs,
}

if _has_torchattacks:
    library_modules["torchattacks"] = torchattacks_configs

if _has_adv_lib:
    library_modules["adv_lib"] = adv_lib_configs

if _has_deeprobust:
    library_modules["deeprobust"] = deeprobust_configs

# Build configuration and getter functions from modules
attack_configs = defaultdict(dict)  # Store config functions
library_getters = defaultdict(dict)  # Store getter functions

for module_name, module in library_modules.items():
    # Gather functions defined in <library>.configs modules
    module_funcs = inspect.getmembers(
        sys.modules[module.__name__],
        predicate=lambda f: inspect.isfunction(f) and f.__module__ == module.__name__,
    )

    for name, func in module_funcs:  # Search for functions that are configs or getters
        config_prefix = module._prefix + "_"
        getter_prefix = "get_" + config_prefix

        if name.startswith(config_prefix) and not name.startswith(getter_prefix):
            # This is a config function (e.g., adv_lib_pgd)
            attack_name = name.removeprefix(config_prefix)
            attack_configs[module_name][attack_name] = func

        elif name.startswith(
            getter_prefix
        ):  # Capture getter function (e.g., get_adv_lib_pgd)
            attack_name = name.removeprefix(getter_prefix)
            library_getters[module_name][attack_name] = func


def list_attacks(threat_model: str = None, lib: str = None) -> list:
    """
    List all available attacks, optionally filtered by threat model and/or library.

    Discovers attacks automatically from registered config modules — no external
    JSON file needed.

    An attack is considered compatible with a threat model if:
      - It has a config function whose ``threat_model`` matches, OR
      - Its getter accepts a ``threat_model`` parameter (multi-norm attack) *and* it has
        a config to take its hyperparameters from for that norm.

    Everything listed here can be built with :func:`get_attack`, so the list can be used
    to drive a sweep without a missing implementation aborting it halfway.

    Args:
        threat_model: Filter by threat model ('l0', 'l1', 'l2', 'linf').
                      If None, returns all attacks.
        lib: Filter by library name ('adv_lib', 'art', 'foolbox', etc.).
             If None, returns attacks from all libraries.

    Returns:
        List of (library, attack_name) tuples for attacks that have a
        corresponding getter function and (optionally) match the requested
        threat model.

    Examples:
        list_attacks()                           # all attacks
        list_attacks(threat_model='linf')        # linf attacks only
        list_attacks(lib='original')             # original library only
        list_attacks('l2', 'foolbox')            # l2 foolbox attacks
    """
    if lib == "torchattacks" and not _has_torchattacks:
        raise ImportError(
            "Torchattacks is not installed. Install attackbenchlib[torchattacks] "
            "in an environment compatible with its requests dependency."
        )
    if lib == "adv_lib" and not _has_adv_lib:
        raise ImportError("adv-lib is not installed; follow the manual install instructions.")
    if lib == "deeprobust" and not _has_deeprobust:
        raise ImportError("DeepRobust is not installed. Install attackbenchlib[deeprobust].")
    results = []
    libs_to_scan = {lib: library_modules[lib]} if lib else library_modules

    for lib_name, module in libs_to_scan.items():
        if lib_name not in library_getters:
            continue

        available_getters = set(library_getters[lib_name].keys())

        if threat_model is None:
            # No threat-model filter: return every attack that has a getter
            for attack_name in sorted(available_getters):
                results.append((lib_name, attack_name))
        else:
            matched_getters = set()

            # 1) Scan config functions whose threat_model matches directly
            for config_name, config_func in attack_configs[lib_name].items():
                cfg_tm = config_func().get("threat_model")

                if cfg_tm != threat_model:
                    continue

                # Map config name back to getter name.  Config names may carry
                # a norm suffix (e.g. "alma_l1") while the getter is just "alma".
                if config_name in available_getters:
                    matched_getters.add(config_name)
                else:
                    for suffix in (f"_{threat_model}", "_NQ"):
                        if config_name.endswith(suffix):
                            candidate = config_name[: -len(suffix)]
                            if candidate in available_getters:
                                matched_getters.add(candidate)
                                break

            # 2) Include multi-norm attacks: a getter that accepts ``threat_model`` works
            #    with any norm it has a config for.
            for getter_name, getter_func in library_getters[lib_name].items():
                sig = inspect.signature(getter_func)
                if (
                    "threat_model" in sig.parameters
                    and _config_for(lib_name, getter_name, threat_model) is not None
                ):
                    matched_getters.add(getter_name)

            for name in sorted(matched_getters):
                if _can_build(lib_name, name, threat_model):
                    results.append((lib_name, name))

    return sorted(results)


@lru_cache(maxsize=None)
def _can_build(lib: str, attack: str, threat_model: str) -> bool:
    """
    Whether get_attack() actually produces this attack for this norm.

    Some getters cover only a subset of the norms (Foolbox's BIM has no l0 variant), and
    a listing that advertises them aborts a sweep halfway through. Building is cheap —
    the getters return partials — and the result is memoised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            get_attack(lib=lib, attack=attack, threat_model=threat_model)
            return True
        except Exception:
            return False


def _config_for(lib: str, attack: str, threat_model: str = None):
    """
    The config function backing (attack, threat_model), or None.

    Config names may carry a norm suffix (``alma_l1``) while the getter is named after
    the attack family (``get_adv_lib_alma``), so an exact-name lookup alone would find
    nothing and leave the attack running on generic defaults instead of its own config.
    """
    configs = attack_configs[lib]

    if threat_model is None:
        return configs.get(attack)

    # A norm-specific config is more precise than the generic family config. In
    # particular, APGD-L1 needs a different epsilon and ``use_largereps`` value,
    # while FMN-Linf uses a different maximum step size.
    specific = configs.get(f"{attack}_{threat_model}")
    if specific is not None and specific().get("threat_model") == threat_model:
        return specific

    # A getter accepting ``threat_model`` only proves that its implementation has a
    # norm switch; it does not make an epsilon or the other hyperparameters meaningful
    # under every norm. A generic config therefore authorises only the norm it
    # explicitly declares.
    generic = configs.get(attack)
    if generic is not None and generic().get("threat_model") == threat_model:
        return generic

    return None


def _is_attack_compatible(lib: str, attack: str, threat_model: str) -> bool:
    """
    Check whether an attack is compatible with a given threat model.

    An attack is considered compatible if:
      - Its getter function accepts a ``threat_model`` parameter (multi-norm), OR
      - There is a config function whose ``threat_model`` matches.
    """
    return _config_for(lib, attack, threat_model) is not None


def get_attack(lib: str, attack: str, threat_model: str, **kwargs) -> Callable:
    """
    Get attack function by library and attack name.

    Args:
        lib: Attack library name ('adv_lib', 'art', 'cleverhans', etc.)
        attack: Attack name within the library ('pgd', 'fgsm', etc.)
        threat_model: Threat model ('l0', 'l1', 'l2', 'linf'). Required.
        **kwargs: Additional parameters to override defaults

    Returns:
        Attack function ready to use

    Raises:
        ValueError: If the attack is not available for the given threat model.

    Examples:
        attack = get_attack(lib='adv_lib', attack='pgd', threat_model='linf')
        attack = get_attack(lib='torchattacks', attack='cw_l2', threat_model='l2')
    """

    if lib == "torchattacks" and not _has_torchattacks:
        raise ImportError(
            "Torchattacks is not installed. Install attackbenchlib[torchattacks] "
            "in an environment compatible with its requests dependency."
        )
    if lib == "adv_lib" and not _has_adv_lib:
        raise ImportError("adv-lib is not installed; follow the manual install instructions.")
    if lib == "deeprobust" and not _has_deeprobust:
        raise ImportError("DeepRobust is not installed. Install attackbenchlib[deeprobust].")
    if lib not in library_getters:
        available_libs = list(library_getters.keys())
        raise ValueError(f"Unknown attack library: {lib}. Available: {available_libs}")
    if attack not in library_getters[lib]:
        available_attacks = list(library_getters[lib].keys())
        raise ValueError(
            f"Unknown attack '{attack}' for library '{lib}'. Available: {available_attacks}"
        )

    # Validate threat model compatibility
    if not _is_attack_compatible(lib, attack, threat_model):
        compatible = [a for _, a in list_attacks(threat_model=threat_model, lib=lib)]
        raise ValueError(
            f"Attack '{attack}' from '{lib}' is not available for threat model '{threat_model}'. "
            f"Compatible attacks for '{lib}' + '{threat_model}': {compatible}"
        )

    # Get the configuration function and getter function
    config_func = _config_for(lib, attack, threat_model)
    getter_func = library_getters[lib][attack]
    sig = inspect.signature(getter_func)

    # Parameters declared by the config function
    params = dict(config_func()) if config_func else {}

    # Override with user-provided kwargs
    params.update(kwargs)

    # The caller asked for this threat model, so it wins over the one the config
    # happens to declare. Without this a multi-norm attack requested for 'l2' was built
    # with the config's norm — list_attacks('l2') then fed a benchmark full of linf
    # attacks labelled l2.
    if "threat_model" in sig.parameters and "threat_model" not in kwargs:
        declared = params.get("threat_model")
        if declared is not None and declared != threat_model and "epsilon" in params:
            warnings.warn(
                f"'{attack}' from '{lib}' has no config for '{threat_model}': using the "
                f"'{declared}' one, so its perturbation budget (epsilon={params['epsilon']}) "
                f"is the one chosen for '{declared}'. Pass epsilon=... for a budget that "
                f"means something under '{threat_model}'."
            )
        params["threat_model"] = threat_model

    # Filter parameters to match function signature
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}

    missing = [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.name != "self"
        and p.name not in filtered_params
    ]
    if missing:
        raise ValueError(
            f"Cannot build '{attack}' from '{lib}' for threat model '{threat_model}': "
            f"no value for {missing}. The config functions for this attack are "
            f"{sorted(n for n in attack_configs[lib] if n == attack or n.startswith(attack + '_'))}"
            f" — add one for '{threat_model}' or pass the parameters explicitly. "
            f"(Guessing them would silently benchmark a differently configured attack.)"
        )

    # Call the getter function
    attack_instance = getter_func(**filtered_params)
    wrapper = library_modules[lib]._wrapper

    if isinstance(attack_instance, dict):
        attack_fn = partial(wrapper, **attack_instance)
    else:
        attack_fn = partial(wrapper, attack=attack_instance)

    # Attach metadata for automatic extraction in run_attack
    attack_fn._attackbench_name = attack
    attack_fn._attackbench_lib = lib

    return attack_fn
