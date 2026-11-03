"""Centralized training configuration for Skinora Acne Severity Models."""
from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BACKEND_DIR / "models"
EXPERIMENTS_DIR = BACKEND_DIR / "experiments"

CLASSES = ["mild", "moderate", "severe", "very_severe"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainingConfig:
    # Experiment metadata
    experiment_id: str = "resnet50_baseline"
    description: str = "ResNet50 transfer learning on ACNE04"
    architecture: str = "resnet50"  # 'resnet50' or 'efficientnet_b0'
    pretrained_weights: str = "IMAGENET1K_V2"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Data & Splits
    val_split_ratio: float = 0.15
    image_size: int = IMAGE_SIZE
    num_workers: int = 0
    extra_data_dir: Optional[str] = None
    split_manifest: Optional[str] = None

    # Hyperparameters
    epochs: int = 8
    batch_size: int = 16
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.2

    # Training Strategy
    # 'frozen': freeze all backbone, train head only
    # 'partial_layer4': freeze layers 1-3, train layer 4 + head
    # 'deep_layer3_4': freeze layers 1-2, train layers 3 & 4 + head with differential lr
    # 'full': unfreeze all with differential lr
    finetune_mode: str = "partial_layer4"

    # Imbalance & Loss
    use_class_weights: bool = True
    use_weighted_sampler: bool = False
    label_smoothing: float = 0.05

    # Optimization & Scheduling
    optimizer_name: str = "AdamW"
    scheduler_name: str = "plateau"  # 'plateau', 'cosine', or 'none'
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 3
    model_selection_metric: str = "val_macro_f1"  # 'val_macro_f1', 'val_loss', 'val_acc'

    # Augmentation
    aug_rotation_degrees: float = 12.0
    aug_horizontal_flip: bool = True
    aug_color_jitter_brightness: float = 0.08
    aug_color_jitter_contrast: float = 0.08
    aug_color_jitter_saturation: float = 0.05
    aug_scale_min: float = 0.85
    aug_scale_max: float = 1.0

    # Paths
    output_dir: Optional[str] = None
    save_best_checkpoint: bool = True
    evaluate_test: bool = False

    def get_output_path(self) -> Path:
        if self.output_dir:
            p = Path(self.output_dir)
        else:
            p = EXPERIMENTS_DIR / self.experiment_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set global seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

