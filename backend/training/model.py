"""Model construction, layer freezing, and checkpoint management."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

from .config import CLASSES, IMAGE_SIZE, TrainingConfig


def build_model(
    config: TrainingConfig, num_classes: int = len(CLASSES)
) -> nn.Module:
    """Build model with custom classification head and layer freezing strategy."""
    arch = config.architecture.lower()

    if arch == "resnet50":
        # Load torchvision pretrained weights (Default is ImageNet1K_V2)
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        except Exception:
            # Offline / local environment fallback
            model = models.resnet50(weights=None)

        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Apply fine-tuning mode
        if config.finetune_mode == "frozen":
            # Only head is trainable
            pass
        elif config.finetune_mode == "partial_layer4":
            # Layer 4 + head trainable
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif config.finetune_mode == "deep_layer3_4":
            # Layer 3, Layer 4 + head trainable
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
        elif config.finetune_mode == "full":
            # All layers trainable
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown finetune_mode: {config.finetune_mode}")

        # Classification Head with optional dropout
        in_features = model.fc.in_features
        if config.dropout_rate > 0.0:
            model.fc = nn.Sequential(
                nn.Dropout(p=config.dropout_rate),
                nn.Linear(in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

        return model

    elif arch == "efficientnet_b0":
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception:
            model = models.efficientnet_b0(weights=None)

        for param in model.parameters():
            param.requires_grad = False

        if config.finetune_mode != "frozen":
            # Train the final blocks (features[7], features[8])
            for block_idx in [7, 8]:
                if block_idx < len(model.features):
                    for param in model.features[block_idx].parameters():
                        param.requires_grad = True

        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        return model

    else:
        raise ValueError(f"Unsupported architecture: {arch}")


def get_parameter_groups(
    model: nn.Module, config: TrainingConfig
) -> List[Dict[str, Any]]:
    """Separate parameter groups for differential learning rates between backbone and head."""
    arch = config.architecture.lower()
    backbone_params = []
    head_params = []

    if arch == "resnet50":
        head_module = model.fc
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc.") or name.startswith("fc"):
                head_params.append(param)
            else:
                backbone_params.append(param)
    elif arch == "efficientnet_b0":
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "classifier" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
    else:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": config.head_lr}]

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": config.backbone_lr,
            "weight_decay": config.weight_decay,
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": config.head_lr,
            "weight_decay": config.weight_decay,
        })
    return groups


def save_checkpoint(
    model: nn.Module,
    config: TrainingConfig,
    metrics: Dict[str, Any],
    filepath: Path,
) -> None:
    """Save model weights and metadata into a reproducible checkpoint."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items()
        },
        "classes": CLASSES,
        "architecture": config.architecture,
        "image_size": config.image_size,
        "finetune_mode": config.finetune_mode,
        "dropout_rate": config.dropout_rate,
        "experiment_id": config.experiment_id,
        "validation_accuracy": metrics.get("val_accuracy"),
        "validation_macro_f1": metrics.get("val_macro_f1"),
        "validation_loss": metrics.get("val_loss"),
        "primary_dataset": "ACNE04",
        "pretraining": config.pretrained_weights,
    }
    torch.save(checkpoint, filepath)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, List[str], Dict[str, Any]]:
    """Safely reconstruct and load a model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint.get("classes", CLASSES)
    arch = checkpoint.get("architecture", "resnet50").lower()
    dropout_rate = checkpoint.get("dropout_rate", 0.0)

    if arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        # Check if saved model used Sequential head
        state_dict = checkpoint["model_state_dict"]
        has_sequential_head = any("fc.1." in k for k in state_dict.keys())
        if has_sequential_head:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, len(classes)),
            )
        else:
            model.fc = nn.Linear(in_features, len(classes))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, len(classes)),
        )
    else:
        raise ValueError(f"Unknown architecture in checkpoint: {arch}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    return model, classes, metadata

