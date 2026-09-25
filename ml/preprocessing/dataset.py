"""
SKINORA ML — PyTorch Dataset & DataLoader

Provides:
    AcneDataset     — reads images from a split CSV + applies transforms
    get_transforms  — returns train/val/test torchvision transform pipelines
    get_dataloaders — returns DataLoader objects for all three splits

Usage:
    from preprocessing.dataset import get_dataloaders
    loaders = get_dataloaders()
    for images, labels in loaders["train"]:
        ...
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_SEED,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def get_transforms() -> dict[str, transforms.Compose]:
    """Return augmented train transforms and deterministic val/test transforms.

    Training augmentations: random flip, rotation, colour jitter.
    Val/Test: only resize + normalize (no randomness).
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AcneDataset(Dataset):
    """PyTorch Dataset for the ACNE04 severity classification task.

    Each row in the CSV must have:
        filename    — image filename (used for display / debugging)
        acne_level  — integer label (0, 1, 2, 3)
        image_path  — absolute or relative path to the image file

    Args:
        dataframe: Pandas DataFrame with the columns above.
        transform: Optional torchvision transform applied to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["acne_level"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> dict[int, int]:
        """Return a mapping of {class_label: count} for this split."""
        return self.data["acne_level"].value_counts().sort_index().to_dict()


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------


def get_dataloaders(
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Load train / val / test CSVs and return DataLoader objects.

    Args:
        num_workers: Number of worker processes for data loading.
                     Use 0 for Windows compatibility (avoids pickling issues).

    Returns:
        Dict with keys "train", "val", "test".
    """
    for csv_path in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                "Run: python -m preprocessing.build_dataset"
            )

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    tfms = get_transforms()

    train_dataset = AcneDataset(train_df, transform=tfms["train"])
    val_dataset = AcneDataset(val_df, transform=tfms["val"])
    test_dataset = AcneDataset(test_df, transform=tfms["test"])

    print(f"Dataset splits:")
    print(f"  Train : {len(train_dataset):>5} images — class counts: {train_dataset.class_counts()}")
    print(f"  Val   : {len(val_dataset):>5} images — class counts: {val_dataset.class_counts()}")
    print(f"  Test  : {len(test_dataset):>5} images — class counts: {test_dataset.class_counts()}")

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        ),
    }

    return loaders
