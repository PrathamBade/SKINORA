"""Unit tests for model construction, layer freezing, and checkpoint serialization."""
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from Skinora.backend.training.config import CLASSES, TrainingConfig
from Skinora.backend.training.model import (
    build_model,
    get_parameter_groups,
    load_model_from_checkpoint,
    save_checkpoint,
)


def test_build_model_frozen():
    config = TrainingConfig(architecture="resnet50", finetune_mode="frozen", dropout_rate=0.2)
    model = build_model(config, num_classes=4)
    assert isinstance(model, nn.Module)
    assert isinstance(model.fc, nn.Sequential)

    # In frozen mode, only fc parameters should require grad
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_build_model_partial_layer4():
    config = TrainingConfig(architecture="resnet50", finetune_mode="partial_layer4", dropout_rate=0.0)
    model = build_model(config, num_classes=4)
    assert isinstance(model.fc, nn.Linear)

    # Layer 4 and fc should require grad
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False


def test_parameter_groups():
    config = TrainingConfig(
        architecture="resnet50",
        finetune_mode="partial_layer4",
        backbone_lr=1e-5,
        head_lr=1e-4,
    )
    model = build_model(config, num_classes=4)
    groups = get_parameter_groups(model, config)
    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-5  # backbone
    assert groups[1]["lr"] == 1e-4  # head


def test_checkpoint_roundtrip():
    config = TrainingConfig(experiment_id="test_exp", architecture="resnet50", finetune_mode="frozen")
    model = build_model(config, num_classes=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_model.pth"
        metrics = {"val_accuracy": 0.75, "val_macro_f1": 0.70, "val_loss": 0.60}
        save_checkpoint(model, config, metrics, ckpt_path)
        assert ckpt_path.exists()

        loaded_model, classes, metadata = load_model_from_checkpoint(ckpt_path)
        assert classes == CLASSES
        assert metadata["validation_accuracy"] == 0.75
        assert metadata["architecture"] == "resnet50"
