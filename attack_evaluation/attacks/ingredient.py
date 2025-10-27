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
    """
    if attack_name:
        # Parse attack_name to extract source and name
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
    
    # Get the configuration function and getter function
    config_func = attack_configs[source].get(name)
    getter_func = library_getters[source][name]
    
    if not config_func:
        print(f"Warning: No configuration function found for attack {name} in source {source}")
        params = {}
    else:
        # Parse parameters from config
        params = _parse_config_params(config_func)
        print(f"Debug: Parsed params from config: {params}")

    # FALLBACK: If parsing fails, use signature inspection
    sig = inspect.signature(getter_func)
    required_params = [p for p in sig.parameters.values() 
                      if p.default == inspect.Parameter.empty and p.name != 'self']
    
    print(f"Debug: Required parameters for {source}_{name}: {[p.name for p in required_params]}")
    
    # Verify that all the required parameters are present
    missing_params = []
    for param in required_params:
        if param.name not in params:
            missing_params.append(param.name)
    
    if missing_params:
        print(f"Warning: Missing required parameters: {missing_params}")
        # Apply default values for missing parameters
        default_values = _get_smart_defaults(source, name, missing_params, threat_model)
        params.update(default_values)
        print(f"Debug: Applied smart defaults: {default_values}")
    
    # Call the getter function with the parameters
    try:
        filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
        
        print(f"Debug: Final params for {source}_{name}: {filtered_params}")
        attack = getter_func(**filtered_params)
        
        wrapper = library_modules[source]._wrapper
        
        if isinstance(attack, dict):
            return partial(wrapper, **attack)
        else:
            return partial(wrapper, attack=attack)
            
    except Exception as e:
        print(f"Error calling getter function for {source}_{name}: {e}")
        print(f"Function signature: {sig}")
        print(f"Provided params: {filtered_params}")
        raise


def _get_smart_defaults(source: str, name: str, missing_params: list, threat_model: str = None) -> dict:
    """Genera valori di default intelligenti per parametri mancanti"""
    defaults = {}
    
    for param_name in missing_params:
        # General defaults based on paramether name
        if param_name in ['eps', 'epsilon']:
            defaults[param_name] = 8/255  # Budget L∞ default
        elif param_name in ['num_steps', 'steps', 'iterations']:
            defaults[param_name] = 40     # default number of iterations
        elif param_name in ['step_size', 'alpha']:
            defaults[param_name] = 2/255  # default step size
        elif param_name in ['relative_step_size']:
            defaults[param_name] = 0.1    # default relative step size
        elif param_name in ['threat_model', 'norm']:
            defaults[param_name] = threat_model or 'linf'
        elif param_name in ['num_random_init', 'random_init']:
            defaults[param_name] = 0
        elif param_name in ['random_eps', 'random_start']:
            defaults[param_name] = False
        elif param_name in ['targeted']:
            defaults[param_name] = False
        elif param_name in ['clip_min']:
            defaults[param_name] = 0.0
        elif param_name in ['clip_max']:
            defaults[param_name] = 1.0
        elif param_name in ['loss_fn', 'loss']:
            defaults[param_name] = None  
        
        
        elif source == 'adv_lib':
            if param_name == 'relative_step_size':
                defaults[param_name] = 0.1 
            elif param_name == 'abs_step_size':
                defaults[param_name] = None
        elif source == 'art':
            if param_name == 'estimator':
                defaults[param_name] = None  
        elif source == 'foolbox':
            if param_name == 'distance':
                defaults[param_name] = 'linf'
        
        # Try none if you can't find a default
        if param_name not in defaults:
            print(f"Warning: No smart default found for parameter '{param_name}', using None")
            defaults[param_name] = None
    
    return defaults


def _parse_config_params(config_func):
    """Parse parameters from configuration function with better handling"""
    params = {}
    
    try:
        source_lines = inspect.getsourcelines(config_func)[0]
        
        for line in source_lines[1:]:  # Skip function definition
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
                
            # Look for variable assignments
            if '=' in line and not line.startswith('#'):
                try:
                    # Split only on first = to handle cases like "x = y = z"
                    var_name, var_value = line.split('=', 1)
                    var_name = var_name.strip()
                    var_value = var_value.strip()
                    
                    # Remove inline comments FIRST
                    if '#' in var_value:
                        var_value = var_value.split('#')[0].strip()
                    
                    # Clean quotes
                    if (var_value.startswith("'") and var_value.endswith("'")) or \
                       (var_value.startswith('"') and var_value.endswith('"')):
                        params[var_name] = var_value[1:-1]
                    # Handle boolean values
                    elif var_value == 'True':
                        params[var_name] = True
                    elif var_value == 'False':
                        params[var_name] = False
                    elif var_value == 'None':
                        params[var_name] = None
                    # Handle numeric values
                    else:
                        try:
                            # Try int first
                            if var_value.isdigit():
                                params[var_name] = int(var_value)
                            # Try float
                            elif '.' in var_value and var_value.replace('.', '').isdigit():
                                params[var_name] = float(var_value)
                            # Try fractions like 8/255
                            elif '/' in var_value:
                                parts = var_value.split('/')
                                if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                                    params[var_name] = float(parts[0]) / float(parts[1])
                        except:
                            continue  # Skip if we can't parse
                except:
                    continue  # Skip malformed lines
    except:
        pass  # If we can't get source, return empty params
    
    return params
