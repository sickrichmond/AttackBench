import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

from . import subsets


class ImageNetKaggle(Dataset):
    def __init__(self, root, split='val', transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.split = split

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        self.samples_lst = os.listdir(samples_dir)

        for entry in self.samples_lst:
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


# Size of the validation subset shipped with the package: the one used in the paper.
PAPER_SUBSET_SIZE = 5000


def prepare_imagenet_subset(root, split='val', n_samples: int = PAPER_SUBSET_SIZE,
                            out_dir: Optional[Union[str, Path]] = None):
    """Draw a fixed list of validation files and write it next to the data (never into
    the installed package, which may be read-only and is shared between runs)."""
    samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
    data_list = np.array(os.listdir(samples_dir))

    np.random.seed(0)
    subset = np.random.choice(data_list, replace=False, size=n_samples)
    out_dir = Path(out_dir) if out_dir is not None else Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'imagenet-{n_samples}-{split}.txt'
    np.savetxt(out_file, subset, fmt='%s')
    return out_file


def load_imagenet(root: Union[str, Path],
                  split: str = 'val',
                  transform=None,
                  n_samples: Optional[int] = PAPER_SUBSET_SIZE) -> Dataset:
    """Full validation set (n_samples=None) or the shipped fixed subset of that size."""
    data = ImageNetKaggle(root=root, split='val', transform=transform)
    if n_samples is None:
        return data

    subset_names_file = Path(os.path.dirname(subsets.__file__)) / f'imagenet-{n_samples}-{split}.txt'
    if not subset_names_file.exists():
        raise FileNotFoundError(
            f"AttackBench only ships the paper's {PAPER_SUBSET_SIZE}-sample validation list "
            f"({subset_names_file.name} not found). For any other size call "
            f"get_loader('imagenet', num_samples=N), which draws a deterministic random "
            f"subset of the full validation set using `seed`."
        )
    subset_names = np.loadtxt(subset_names_file, dtype=str)
    subset_indices = [i for i, file in enumerate(data.samples_lst) if file in subset_names]
    return Subset(dataset=data, indices=subset_indices)
