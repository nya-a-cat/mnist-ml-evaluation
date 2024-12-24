import os
from typing import Tuple, Dict, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTDataset(Dataset):
    """
    A custom Dataset wrapper for MNIST that formats data for the CapsuleNet model.
    Handles data from the project's data directory structure.
    """

    def __init__(self, dataset: datasets.MNIST):
        """
        Initialize the dataset wrapper.

        Args:
            dataset: The MNIST dataset from torchvision
        """
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image, label = self.dataset[idx]
        return {
            "pixel_values": image,
            "labels": label
        }


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get the transforms for MNIST dataset.

    Args:
        train: Whether to get transforms for training or evaluation

    Returns:
        transforms.Compose: The composition of transforms
    """
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if train:
        # Add any training-specific transforms here if needed
        pass

    return transforms.Compose(transform_list)


def load_mnist_data(
        data_dir: str = os.path.join('data', 'MNIST'),
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare MNIST datasets with DataLoaders.
    Follows the project's directory structure for data management.

    Args:
        data_dir: Directory to store the dataset (relative to project root)
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        pin_memory: If True, pin memory for faster data transfer to GPU

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )

    # Load validation data
    val_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )

    # Wrap datasets
    train_dataset = MNISTDataset(train_dataset)
    val_dataset = MNISTDataset(val_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader


def get_dataset_info() -> Dict[str, int]:
    """
    Get information about the MNIST dataset.

    Returns:
        dict: Dictionary containing dataset information
    """
    return {
        "num_classes": 10,
        "input_channels": 1,
        "input_height": 28,
        "input_width": 28,
        "train_samples": 60000,
        "test_samples": 10000
    }


if __name__ == "__main__":
    # Simple test to verify data loading works
    import sys
    import pathlib

    # Add project root to path to allow imports from parent directory
    project_root = pathlib.Path(__file__).parent.parent
    sys.path.append(str(project_root))

    # Test data loading
    train_loader, val_loader = load_mnist_data()
    dataset_info = get_dataset_info()

    print("Data loading test:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Dataset info: {dataset_info}")

    # Test a single batch
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {batch['pixel_values'].shape}")
    print(f"Labels: {batch['labels'].shape}")