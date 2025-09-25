import inspect
import sys
from collections import defaultdict
from functools import partial
from typing import Callable

from .adv_lib import configs as adv_lib_configs
from .art import configs as art_configs
from .cleverhans import configs as cleverhans_configs
from .deeprobust import configs as deeprobust_configs
from .foolbox import configs as foolbox_configs
from .original import configs as original_configs
from .torchattacks import configs as torchattacks_configs

library_modules = {
    'adv_lib': adv_lib_configs,
    'art': art_configs,
    'cleverhans': cleverhans_configs,
    'deeprobust': deeprobust_configs,
    'foolbox': foolbox_configs,
    'original': original_configs,
    'torchattacks': torchattacks_configs,
}

# Build getters dictionary from config modules
library_getters = defaultdict(dict)

# Build configuration and getter functions from modules
attack_configs = defaultdict(dict)  # Store config functions
library_getters = defaultdict(dict)  # Store getter functions

for module_name, module in library_modules.items():
    # gather function defined in <library>.configs modules
    module_funcs = inspect.getmembers(sys.modules[module.__name__],
                                     predicate=lambda f: inspect.isfunction(f) and f.__module__ == module.__name__)

    for name, func in module_funcs:  # search for functions that are configs or getters
        config_prefix = module._prefix + '_'
        getter_prefix = 'get_' + config_prefix

        if name.startswith(config_prefix) and not name.startswith(getter_prefix):
            # This is a config function (e.g., adv_lib_pgd)
            attack_name = name.removeprefix(config_prefix)
            attack_configs[module_name][attack_name] = func
            
        elif name.startswith(getter_prefix):  # capture getter function (e.g., get_adv_lib_pgd)
            attack_name = name.removeprefix(getter_prefix)
            library_getters[module_name][attack_name] = func


def get_attack(attack_name: str = None, source: str = None, name: str = None, threat_model: str = None) -> Callable:
    """
    Get attack function by name or source/name combination.
    
    Args:
        attack_name: Full attack name (e.g., 'adv_lib_pgd', 'original_autopgd')
        source: Library source (e.g., 'adv_lib', 'original') 
        name: Attack name within the source (e.g., 'pgd', 'autopgd')
        threat_model: Threat model (e.g., 'l2', 'linf')
    
    Returns:
        Callable attack function
    """
    if attack_name:
        # Parse attack_name to extract source and name
        # Try to match against known library sources first
        source = None
        name = None
        
        for lib_source in library_getters.keys():
            if attack_name.startswith(lib_source + '_'):
                source = lib_source
                name = attack_name[len(lib_source) + 1:]  # Remove source + '_'
                break
        
        if not source:
            raise ValueError(f"Invalid attack_name format: {attack_name}. Must start with one of: {list(library_getters.keys())}")
    
    if not source or not name:
        raise ValueError("Must provide either attack_name or both source and name")
    
    if source not in library_getters:
        raise ValueError(f"Unknown attack source: {source}. Available: {list(library_getters.keys())}")
    
    if name not in library_getters[source]:
        raise ValueError(f"Unknown attack name '{name}' for source '{source}'. Available: {list(library_getters[source].keys())}")
    
    if source not in library_getters:
        raise ValueError(f"Unknown attack source: {source}. Available: {list(library_getters.keys())}")
    
    if name not in library_getters[source]:
        raise ValueError(f"Unknown attack name '{name}' for source '{source}'. Available: {list(library_getters[source].keys())}")
    
    # Get the configuration function and getter function
    config_func = attack_configs[source].get(name)
    getter_func = library_getters[source][name]
    
    if not config_func:
        raise ValueError(f"No configuration function found for attack {name} in source {source}")
    
    # Simple approach: extract parameters by examining the source code
    source_lines = inspect.getsourcelines(config_func)[0]
    
    # Parse the function body to extract variable assignments
    params = {}
    for line in source_lines[1:]:  # Skip the function definition line
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            try:
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()
                var_value = var_value.strip()
                
                # Simple evaluation of common values
                if var_value == 'True':
                    params[var_name] = True
                elif var_value == 'False':
                    params[var_name] = False
                elif var_value == 'None':
                    params[var_name] = None
                elif var_value.startswith("'") and var_value.endswith("'"):
                    params[var_name] = var_value[1:-1]  # String literal
                elif var_value.startswith('"') and var_value.endswith('"'):
                    params[var_name] = var_value[1:-1]  # String literal
                else:
                    # Try to evaluate as a Python expression safely
                    try:
                        # Safe evaluation for numeric expressions
                        if all(c in '0123456789.+-*/ ()' for c in var_value):
                            params[var_name] = eval(var_value)
                        elif var_value.isdigit():
                            params[var_name] = int(var_value)
                        elif var_value.replace('.', '').isdigit():
                            params[var_name] = float(var_value)
                    except:
                        continue  # Skip if we can't evaluate
            except:
                continue  # Skip lines we can't parse
    
    # Call the getter function with the parameters
    try:
        sig = inspect.signature(getter_func)
        filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
        attack = getter_func(**filtered_params)
        
        wrapper = library_modules[source]._wrapper
        
        if isinstance(attack, dict):
            return partial(wrapper, **attack)
        else:
            return partial(wrapper, attack=attack)
            
    except Exception as e:
        print(f"Error calling getter function for {source}_{name}: {e}")
        print(f"Parsed params: {params}")
        print(f"Filtered params: {filtered_params}")
        raise
