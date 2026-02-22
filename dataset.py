"""
CIFAR-10 Laden, 80/20-Split und Subset-Erzeugung.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

import config


def get_transforms(train: bool = True) -> transforms.Compose:
    """Data Augmentation und Normalisierung."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.CIFAR10_MEAN, config.CIFAR10_STD),
        ])


def load_cifar10():
    """Lädt CIFAR-10 und gibt Train/Test-Datasets zurück."""
    train_dataset = torchvision.datasets.CIFAR10(
        root=config.DATASET_ROOT, train=True, download=True,
        transform=get_transforms(train=True),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=config.DATASET_ROOT, train=False, download=True,
        transform=get_transforms(train=False),
    )
    return train_dataset, test_dataset


def split_train_val(train_dataset, seed: int = config.BASE_SEED):
    """
    Teilt die 60.000 Trainingsbilder in 48.000 Train und 12.000 Validation.
    Gibt die Indizes zurück (nicht die Datasets selbst), damit Subsets
    unabhängig von der Augmentation-Transformation erzeugt werden können.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(train_dataset))
    rng.shuffle(indices)

    train_indices = indices[:config.TRAIN_SIZE]
    val_indices = indices[config.TRAIN_SIZE:config.TRAIN_SIZE + config.VAL_SIZE]

    return train_indices, val_indices


def get_subset_indices(train_indices: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Wählt n zufällige Indizes aus train_indices für Partition k.
    Verwendet deterministischen Seed basierend auf (n, k).
    """
    seed = config.get_seed(n, k)
    rng = np.random.RandomState(seed)

    if n >= len(train_indices):
        return train_indices.copy()

    chosen = rng.choice(len(train_indices), size=n, replace=False)
    return train_indices[chosen]


def create_dataloaders(
    train_dataset,
    test_dataset,
    val_dataset,  # NEU: als Parameter übergeben statt neu laden!
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    subset_indices: np.ndarray,
) -> tuple:
    """
    Erstellt DataLoader für Training-Subset, Validation und Test.
    """
    train_subset = Subset(train_dataset, subset_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=False,
    )

    return train_loader, val_loader, test_loader
