"""
SKINORA ML — Model Architecture

Builds a ResNet18 classifier for acne severity (4 classes).

Strategy:
    - Load pretrained ResNet18 (ImageNet weights)
    - Freeze all backbone layers
    - Replace the final FC layer with a 4-class head
    - Only the classifier head is trained in Phase 2 (fast, CPU-friendly)

In later phases, the backbone can be unfrozen for fine-tuning.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.config import FREEZE_BACKBONE, NUM_CLASSES, PRETRAINED


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    freeze_backbone: bool = FREEZE_BACKBONE,
) -> nn.Module:
    """Build and return a ResNet18 acne severity classifier.

    Args:
        num_classes: Number of output classes (default 4).
        pretrained: If True, load ImageNet pretrained weights.
        freeze_backbone: If True, freeze all layers except the classifier.

    Returns:
        Configured nn.Module ready for training.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    # The new fc layer has requires_grad=True by default

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet18 | "
          f"Trainable params: {trainable:,} / {total:,} "
          f"({'frozen backbone' if freeze_backbone else 'full fine-tune'})")

    return model
