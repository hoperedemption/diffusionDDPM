import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class CIFAR10Dataset:
    """
    Manages loading and preprocessing of the CIFAR-10 dataset.
    """
    def __init__(self, image_size: int = 32, data_dir: str = './data'):
        """
        Initializes the CIFAR10Dataset.

        Args:
            image_size: The desired square size for image resizing.
            data_dir: Directory where CIFAR-10 data will be downloaded.
        """
        self.image_size = image_size
        self.data_dir = data_dir
        self.transform = self._get_transform()

    def _get_transform(self) -> transforms.Compose:
        """
        Defines the image transformations for CIFAR-10.
        Images are resized, converted to tensor, and normalized to [-1, 1].

        Returns:
            A torchvision.transforms.Compose object.
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # Normalize to [-1, 1]
        ])

    def get_train_dataloader(self, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        """
        Returns a DataLoader for the CIFAR-10 training set.

        Args:
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory.

        Returns:
            A torch.utils.data.DataLoader for the training set.
        """
        train_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0 # Keep workers alive between epochs
        )

    def get_test_dataloader(self, batch_size: int, num_workers: int, pin_memory: bool) -> DataLoader:
        """
        Returns a DataLoader for the CIFAR-10 test set (typically used for evaluation/sampling in DDPM).

        Args:
            batch_size: Number of samples per batch.
            num_workers: Number of subprocesses to use for data loading.
            pin_memory: If True, the data loader will copy Tensors into CUDA pinned memory.

        Returns:
            A torch.utils.data.DataLoader for the test set.
        """
        test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        return DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )