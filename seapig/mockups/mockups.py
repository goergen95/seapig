"""Supplies mockup classes to be used in testing and notebooks."""

import random
from typing import override

import torch
from torch.utils.data import Dataset, Subset


class MockupDataset(Dataset):  # type: ignore [type-arg]
    """Supplies a mockup PyTorch Dataset."""

    images: torch.Tensor
    labels: torch.Tensor
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]
    return_dict: bool

    def __init__(self, n_samples: int = 1000, return_dict: bool = True) -> None:
        super().__init__()
        self.images = torch.rand((n_samples, 3, 128, 128))
        self.labels = torch.randint_like(self.images, low=0, high=2)
        ids = list(range(0, n_samples))
        random.shuffle(ids)
        k = int(n_samples / 3)
        self.train_indices = ids[:k]
        self.val_indices = ids[k : int(2 * k)]
        self.test_indices = ids[int(2 * k) :]
        self.return_dict = return_dict

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.images.shape[0]

    @override
    def __getitem__(self, index: int):  # type: ignore [no-untyped-def]
        """Return a single sample from the dataset."""
        if self.return_dict:
            return {
                "inputs": self.images[index, :],
                "labels": self.labels[index, :],
            }
        else:
            return self.images[index, :]

    def get_train_split(self):  # type: ignore [no-untyped-def]
        """Return a subset based on the training indices."""
        return Subset(dataset=self, indices=self.train_indices)

    def get_val_split(self):  # type: ignore [no-untyped-def]
        """Return a subset based on the validation indices."""
        return Subset(dataset=self, indices=self.val_indices)

    def get_test_split(self):  # type: ignore [no-untyped-def]
        """Return a subset based on the testing indices."""
        return Subset(dataset=self, indices=self.test_indices)


class MockupCNN(torch.nn.Module):
    """Supplies a mockup CNN-Model."""

    layer1: torch.nn.Module
    layer2: torch.nn.Module
    avg: bool

    def __init__(self, avg: bool = True) -> None:
        super().__init__()
        self.avg = avg
        self.layer1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3
        )
        self.layer2 = torch.nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=3
        )

    def forward(self, x: torch.Tensor):  # type: ignore [no-untyped-def]
        """Implement the forward method."""
        return self.layer2(self.layer1(x))

    def embed(self, x: torch.Tensor):  # type: ignore [no-untyped-def]
        """Implement an embed method."""
        out = self.layer1(x)
        if self.avg:
            out = out.view(out.size()[0], out.size()[1], -1).mean(2)
        return out
