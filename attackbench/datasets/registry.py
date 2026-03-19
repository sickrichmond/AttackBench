from pathlib import Path
from typing import Optional

import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST

from .imagenet import load_imagenet


def get_mnist(root: str = 'data') -> Dataset:
    """Get MNIST dataset"""
    transform = transforms.ToTensor()
    dataset = MNIST(root=root, train=False, transform=transform, download=True)
    return dataset


def get_cifar10(root: str = 'data') -> Dataset:
    """Get CIFAR-10 dataset"""
    transform = transforms.ToTensor()
    dataset = CIFAR10(root=root, train=False, transform=transform, download=True)
    return dataset


def get_imagenet(root: str = 'data', num_samples: Optional[int] = None) -> Dataset:
    """Get ImageNet dataset"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    data_path = Path(root) / 'imagenet-data'
    dataset = load_imagenet(root=data_path, split='val', transform=transform, n_samples=num_samples)
    print("Dataset size: ", len(dataset))
    return dataset


_datasets = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'imagenet': get_imagenet,
}


def get_dataset(dataset: str, root: str = 'data', num_samples: Optional[int] = None):
    """Get dataset by name"""
    if dataset not in _datasets:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(_datasets.keys())}")
    
    dataset_func = _datasets[dataset]
    if dataset == 'imagenet':
        return dataset_func(root=root, num_samples=num_samples)
    else:
        return dataset_func(root=root)


def get_loader(dataset: str, batch_size: int = 128, num_samples: Optional[int] = None,
               random_subset: bool = True, seed: int = 0, root: str = 'data') -> DataLoader:
    """Get data loader for specified dataset.
    
    When random_subset=True and num_samples is specified, a deterministic random
    subset is selected using the given seed. This ensures the same (dataset, num_samples, seed)
    always returns the same samples, which is critical for reproducible benchmarking.
    """
    data = get_dataset(dataset=dataset, root=root, num_samples=num_samples)

    if num_samples is not None and num_samples < len(data):
        if not random_subset:
            data = Subset(data, indices=list(range(num_samples)))
        else:
            rng = np.random.default_rng(seed=seed)
            indices = rng.choice(len(data), replace=False, size=num_samples)
            data = Subset(data, indices=indices.tolist())
    
    loader = DataLoader(dataset=data, batch_size=batch_size)
    
    # Attach metadata for automatic extraction in run_attack
    loader._attackbench_dataset = dataset
    
    return loader
