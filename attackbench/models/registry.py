import os
import urllib.request
from functools import partial

import torch
from torch import nn

from .benchmodel_wrapper import BenchModel
from .mnist import SmallCNN
from .original.utils import load_original_model


# Base URL for downloading MNIST checkpoints from GitHub Releases
_CHECKPOINT_BASE_URL = 'https://github.com/sickrichmond/AttackBench/releases/download/checkpoints'

_MNIST_CHECKPOINTS = {
    'mnist_smallcnn_standard.pth': f'{_CHECKPOINT_BASE_URL}/mnist_smallcnn_standard.pth',
    'mnist_smallcnn_robust_ddn.pth': f'{_CHECKPOINT_BASE_URL}/mnist_smallcnn_robust_ddn.pth',
    'mnist_smallcnn_robust_trades.pth': f'{_CHECKPOINT_BASE_URL}/mnist_smallcnn_robust_trades.pth',
}

# Model configurations - maps model names to their parameters
MODEL_CONFIGS = {
    'mnist_smallcnn': {
        'name': 'MNIST_SmallCNN',
        'dataset': 'mnist',
        'source': 'local'
    },
    'mnist_smallcnn_ddn': {
        'name': 'MNIST_SmallCNN_ddn', 
        'dataset': 'mnist',
        'source': 'local'
    },
    'mnist_smallcnn_trades': {
        'name': 'MNIST_SmallCNN_trades',
        'dataset': 'mnist', 
        'source': 'local'
    },
    'carmon_2019': {
        'name': 'Carmon2019Unlabeled',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'chen_2020': {
        'name': 'Chen2020Adversarial',
        'source': 'robustbench',
        'dataset': 'cifar10', 
        'threat_model': 'Linf'
    },
    'debenedetti_2022': {
        'name': 'Debenedetti2022Light_XCiT-S12',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf'
    },
    # Add more configurations as needed...
}


def get_mnist_smallcnn(checkpoint: str) -> nn.Module:
    """Load MNIST SmallCNN model with specified checkpoint, downloading if needed."""
    model = SmallCNN()
    cache_dir = os.path.join('models', 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    model_file = os.path.join(cache_dir, checkpoint)
    if not os.path.exists(model_file):
        if checkpoint not in _MNIST_CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint: {checkpoint}. Available: {list(_MNIST_CHECKPOINTS.keys())}")
        print(f'Downloading {checkpoint}...')
        urllib.request.urlretrieve(_MNIST_CHECKPOINTS[checkpoint], model_file)
    state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    return model


_local_models = {
    'MNIST_SmallCNN': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_standard.pth'),
    'MNIST_SmallCNN_ddn': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_robust_ddn.pth'),
    'MNIST_SmallCNN_trades': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_robust_trades.pth'),
}


def get_local_model(name: str, dataset: str) -> nn.Module:
    """Get local model by name"""
    if name not in _local_models:
        raise ValueError(f"Unknown local model: {name}. Available: {list(_local_models.keys())}")
    return _local_models[name]()


# Add more model configurations to MODEL_CONFIGS above as needed
MODEL_CONFIGS.update({
    'augustin_2020': {
        'name': 'Augustin2020Adversarial',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'L2'
    },
    'standard': {
        'name': 'Standard',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'engstrom_2019': {
        'name': 'Engstrom2019Robustness',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'L2'
    },
    'gowal_2021': {
        'name': 'Gowal2021Improving_70_16_ddpm_100m',
        'source': 'robustbench',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'salman_2020': {
        'name': 'Salman2020Do_50_2',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf'
    },
    'wong_2020': {
        'name': 'Wong2020Fast',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf'
    },
    'standard_imagenet': {
        'name': 'Standard_R50',
        'source': 'robustbench',
        'dataset': 'imagenet',
        'threat_model': 'Linf'
    },
    'stutz_2020': {
        'name': 'Stutz2020CCAT',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'zhang_2020_large': {
        'name': 'Zhang2020CrownLarge',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'zhang_2020_small': {
        'name': 'Zhang2020CrownSmall',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'xiao_2020': {
        'name': 'Xiao2020KWTA',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'wang_2023_small': {
        'name': 'Wang2023DMAdvSmall',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    },
    'wang_2023_large': {
        'name': 'Wang2023DMAdvLarge',
        'source': 'original',
        'dataset': 'cifar10',
        'threat_model': 'Linf'
    }
})


def get_robustbench_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    """Get model from RobustBench"""
    try:
        from robustbench import load_model
    except ModuleNotFoundError as e:
        if 'autoattack' in str(e):
            raise ImportError(
                "robustbench requires 'autoattack', which failed to install from PyPI.\n"
                "Fix with: pip install git+https://github.com/fra31/auto-attack"
            ) from e
        raise ImportError(
            "robustbench is required to load RobustBench models. "
            "Install it with: pip install attackbenchlib[models]"
        ) from e
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


def get_original_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    """Get original model"""
    model = load_original_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


_model_getters = {
    'local': get_local_model,
    'robustbench': get_robustbench_model,
    'original': get_original_model
}


def get_model(model_name: str = None, dataset: str = None, source: str = 'local', 
              requires_grad: bool = False, enforce_box: bool = True, 
              num_max_propagations: int = None) -> BenchModel:
    """
    Get model wrapped in BenchModel.
    
    Args:
        model_name: Name of the model configuration
        dataset: Dataset name (if not using model_name)
        source: Model source ('local', 'robustbench', 'original')
        requires_grad: Whether model requires gradients
        enforce_box: Whether to enforce box constraint [0,1]
        num_max_propagations: Max forward/backward propagations
    
    Returns:
        BenchModel wrapped model
    """
    if model_name and model_name in MODEL_CONFIGS:
        config = MODEL_CONFIGS[model_name]
        source = config['source']
        dataset = config['dataset']
        threat_model = config.get('threat_model', 'Linf')
        model_name_internal = config['name']
    else:
        # Direct specification
        model_name_internal = model_name
        threat_model = 'Linf'  # default
    
    if source not in _model_getters:
        raise ValueError(f"Unknown model source: {source}. Available: {list(_model_getters.keys())}")
    
    if source == 'local':
        base_model = _model_getters[source](model_name_internal, dataset)
    else:
        base_model = _model_getters[source](model_name_internal, dataset, threat_model)
    
    model = BenchModel(base_model, enforce_box=enforce_box, num_max_propagations=num_max_propagations)
    model.eval()
    model.requires_grad_(requires_grad)
    
    # Attach metadata for automatic extraction in run_attack
    model._attackbench_model = model_name_internal
    
    return model
