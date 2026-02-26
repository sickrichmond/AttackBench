import inspect
import sys
from collections import defaultdict
from functools import partial
from typing import Callable

from .art import configs as art_configs
from .cleverhans import configs as cleverhans_configs
from .foolbox import configs as foolbox_configs
from .original import configs as original_configs
from .torchattacks import configs as torchattacks_configs

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
    'art': art_configs,
    'cleverhans': cleverhans_configs,
    'foolbox': foolbox_configs,
    'original': original_configs,
    'torchattacks': torchattacks_configs,
}

if _has_adv_lib:
    library_modules['adv_lib'] = adv_lib_configs

if _has_deeprobust:
    library_modules['deeprobust'] = deeprobust_configs

# Build configuration and getter functions from modules
attack_configs = defaultdict(dict)  # Store config functions
library_getters = defaultdict(dict)  # Store getter functions

for module_name, module in library_modules.items():
    # Gather functions defined in <library>.configs modules
    module_funcs = inspect.getmembers(sys.modules[module.__name__],
                                     predicate=lambda f: inspect.isfunction(f) and f.__module__ == module.__name__)

    for name, func in module_funcs:  # Search for functions that are configs or getters
        config_prefix = module._prefix + '_'
        getter_prefix = 'get_' + config_prefix

        if name.startswith(config_prefix) and not name.startswith(getter_prefix):
            # This is a config function (e.g., adv_lib_pgd)
            attack_name = name.removeprefix(config_prefix)
            attack_configs[module_name][attack_name] = func
            
        elif name.startswith(getter_prefix):  # Capture getter function (e.g., get_adv_lib_pgd)
            attack_name = name.removeprefix(getter_prefix)
            library_getters[module_name][attack_name] = func


def get_attack(lib: str, attack: str, threat_model: str = None, **kwargs) -> Callable:
    """
    Get attack function by library and attack name.
    
    Args:
        lib: Attack library name ('adv_lib', 'art', 'cleverhans', etc.)
        attack: Attack name within the library ('pgd', 'fgsm', etc.)
        threat_model: Threat model for smart defaults
        **kwargs: Additional parameters to override defaults
        
    Returns:
        Attack function ready to use
        
    Examples:
        attack = get_attack(lib='adv_lib', attack='pgd', threat_model='linf')
        attack = get_attack(lib='torchattacks', attack='fgsm')
    """
    
    if lib not in library_getters:
        available_libs = list(library_getters.keys())
        raise ValueError(f"Unknown attack library: {lib}. Available: {available_libs}")
    
    if attack not in library_getters[lib]:
        available_attacks = list(library_getters[lib].keys())
        raise ValueError(f"Unknown attack '{attack}' for library '{lib}'. Available: {available_attacks}")
    
    # Get the configuration function and getter function
    config_func = attack_configs[lib].get(attack)
    getter_func = library_getters[lib][attack]
    
    # Parse parameters from config
    if config_func:
        params = _parse_config_params(config_func)
    else:
        print(f"Warning: No configuration function found for {lib}_{attack}")
        params = {}

    # Check for missing required parameters
    sig = inspect.signature(getter_func)
    required_params = [p for p in sig.parameters.values() 
                      if p.default == inspect.Parameter.empty and p.name != 'self']
    
    # Apply smart defaults for missing parameters
    missing_params = [p.name for p in required_params if p.name not in params]
    if missing_params:
        default_values = _get_smart_defaults(lib, attack, missing_params, threat_model)
        params.update(default_values)
    
    # Override with user-provided kwargs
    params.update(kwargs)
    
    # Filter parameters to match function signature
    filtered_params = {k: v for k, v in params.items() if k in sig.parameters}
    
    # Call the getter function
    try:
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
            
    except Exception as e:
        print(f"Error creating attack {lib}_{attack}: {e}")
        print(f"Function signature: {sig}")
        print(f"Provided params: {filtered_params}")
        raise


def _get_smart_defaults(lib: str, attack: str, missing_params: list, threat_model: str = None) -> dict:
    """Generate smart default values for missing parameters"""
    defaults = {}
    
    for param_name in missing_params:
        # General defaults based on parameter name
        if param_name in ['eps', 'epsilon']:
            defaults[param_name] = 8/255  # Default L-infinity budget
        elif param_name in ['num_steps', 'steps', 'iterations']:
            defaults[param_name] = 40     # Default number of iterations
        elif param_name in ['step_size', 'alpha']:
            defaults[param_name] = 2/255  # Default step size
        elif param_name in ['relative_step_size']:
            defaults[param_name] = 0.1    # Default relative step size
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
        
        # Library-specific defaults
        elif lib == 'adv_lib':
            if param_name == 'relative_step_size':
                defaults[param_name] = 0.1 
            elif param_name == 'abs_step_size':
                defaults[param_name] = None
        elif lib == 'art':
            if param_name == 'estimator':
                defaults[param_name] = None  
        elif lib == 'foolbox':
            if param_name == 'distance':
                defaults[param_name] = 'linf'
        
        # Fallback to None if no smart default found
        if param_name not in defaults:
            print(f"Warning: No smart default found for parameter '{param_name}', using None")
            defaults[param_name] = None
    
    return defaults


def _parse_config_params(config_func):
    """Parse parameters from configuration function"""
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
                    
                    # Remove inline comments
                    if '#' in var_value:
                        var_value = var_value.split('#')[0].strip()
                    
                    # Parse different value types
                    if (var_value.startswith("'") and var_value.endswith("'")) or \
                       (var_value.startswith('"') and var_value.endswith('"')):
                        params[var_name] = var_value[1:-1]
                    elif var_value == 'True':
                        params[var_name] = True
                    elif var_value == 'False':
                        params[var_name] = False
                    elif var_value == 'None':
                        params[var_name] = None
                    else:
                        try:
                            if var_value.isdigit():
                                params[var_name] = int(var_value)
                            elif '.' in var_value and var_value.replace('.', '').isdigit():
                                params[var_name] = float(var_value)
                            elif '/' in var_value:
                                parts = var_value.split('/')
                                if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                                    params[var_name] = float(parts[0]) / float(parts[1])
                        except:
                            continue
                except:
                    continue
    except:
        pass
    
    return params
