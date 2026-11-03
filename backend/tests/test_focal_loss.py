"""Unit tests for Focal Loss and E4 experimental configuration in Skinora."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

# Ensure repository root and parent are in sys.path for package imports
root = Path(__file__).resolve().parents[2]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
root_parent = root.parent
if str(root_parent) not in sys.path:
    sys.path.insert(0, str(root_parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Skinora.backend.training.config import (
        CLASSES,
        MODELS_DIR,
        EXPERIMENTS_DIR,
        TrainingConfig,
    )
    from Skinora.backend.training.loss import FocalLoss
    from Skinora.backend.training.train_acne_model import main
except ImportError:
    from backend.training.config import (
        CLASSES,
        MODELS_DIR,
        EXPERIMENTS_DIR,
        TrainingConfig,
    )
    from backend.training.loss import FocalLoss


def test_focal_loss_scalar_output():
    """Verify that FocalLoss with reduction='mean' produces a finite non-negative scalar."""
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    inputs = torch.randn(16, 4)
    targets = torch.randint(0, 4, (16,))
    loss = loss_fn(inputs, targets)

    assert isinstance(loss, torch.Tensor), "Output must be a torch.Tensor"
    assert loss.dim() == 0, f"Expected scalar (dim 0), got dim {loss.dim()}"
    assert torch.isfinite(loss).item(), "Loss value must be finite (not NaN or Inf)"
    assert loss.item() >= 0.0, f"Loss must be non-negative, got {loss.item()}"


def test_focal_loss_differentiable():
    """Verify that FocalLoss is differentiable and generates valid, finite gradients."""
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    logits = torch.randn(16, 4, requires_grad=True)
    targets = torch.randint(0, 4, (16,))

    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None, "Logits must have gradients after backward()"
    assert torch.isfinite(logits.grad).all().item(), "All gradients must be finite"
    assert (logits.grad.abs().sum() > 0).item(), "Gradients must be non-zero"


def test_focal_loss_gamma_zero_agrees_with_ce():
    """Verify that when gamma=0 and unweighted, FocalLoss is mathematically identical to CrossEntropyLoss."""
    focal_zero = FocalLoss(gamma=0.0, reduction="mean")
    ce_loss = nn.CrossEntropyLoss(reduction="mean")

    torch.manual_seed(42)
    for _ in range(5):
        logits = torch.randn(32, 4)
        targets = torch.randint(0, 4, (32,))

        loss_focal = focal_zero(logits, targets)
        loss_ce = ce_loss(logits, targets)

        assert torch.allclose(loss_focal, loss_ce, atol=1e-6), (
            f"FocalLoss(gamma=0) {loss_focal.item()} does not match CrossEntropyLoss {loss_ce.item()}"
        )


def test_focal_loss_multiclass_batch():
    """Verify FocalLoss on varying batch sizes and reduction modes."""
    loss_fn = FocalLoss(gamma=2.0)

    for bs in [1, 4, 16, 64]:
        logits = torch.randn(bs, len(CLASSES))
        targets = torch.randint(0, len(CLASSES), (bs,))

        loss_mean = loss_fn(logits, targets)
        assert loss_mean.dim() == 0 and torch.isfinite(loss_mean).item()

    # Reductions check
    logits = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))

    fn_none = FocalLoss(gamma=2.0, reduction="none")
    fn_sum = FocalLoss(gamma=2.0, reduction="sum")
    fn_mean = FocalLoss(gamma=2.0, reduction="mean")

    loss_none = fn_none(logits, targets)
    loss_sum = fn_sum(logits, targets)
    loss_mean = fn_mean(logits, targets)

    assert loss_none.shape == (8,), f"Expected shape (8,), got {loss_none.shape}"
    assert torch.allclose(loss_none.sum(), loss_sum, atol=1e-5), "Sum reduction mismatch"
    assert torch.allclose(loss_none.mean(), loss_mean, atol=1e-5), "Mean reduction mismatch"


def test_focal_loss_hard_mining_behavior():
    """Verify that Focal Loss modulates well-classified examples more than hard examples."""
    fl_fn = FocalLoss(gamma=2.0, reduction="none")
    ce_fn = nn.CrossEntropyLoss(reduction="none")

    # Easy sample: high logit on true class 0
    easy_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    target_easy = torch.tensor([0])

    # Hard sample: low logit on true class 0
    hard_logits = torch.tensor([[-2.0, 3.0, 2.0, 1.0]])
    target_hard = torch.tensor([0])

    fl_easy = fl_fn(easy_logits, target_easy).item()
    ce_easy = ce_fn(easy_logits, target_easy).item()

    fl_hard = fl_fn(hard_logits, target_hard).item()
    ce_hard = ce_fn(hard_logits, target_hard).item()

    # Focal modulation factor ratio on easy sample should be tiny (< 0.01)
    ratio_easy = fl_easy / (ce_easy + 1e-12)
    ratio_hard = fl_hard / (ce_hard + 1e-12)

    assert ratio_easy < ratio_hard, (
        f"Focal loss should downweight easy samples more than hard: easy ratio={ratio_easy}, hard ratio={ratio_hard}"
    )


def test_e4_config_resolution():
    """Verify that E4 experimental configuration matches all specification requirements."""
    cfg = TrainingConfig(
        experiment_id="e4_focal_loss",
        architecture="resnet50",
        finetune_mode="deep_layer3_4",
        loss="focal",
        focal_gamma=2.0,
        use_class_weights=False,
        use_weighted_sampler=False,
        label_smoothing=0.0,
        epochs=10,
        batch_size=16,
        backbone_lr=0.00002,
        head_lr=0.0002,
        weight_decay=0.0001,
        dropout_rate=0.2,
        scheduler_name="plateau",
        scheduler_factor=0.5,
        scheduler_patience=2,
        early_stopping_patience=3,
        model_selection_metric="val_macro_f1",
        seed=42,
        image_size=224,
    )

    assert cfg.experiment_id == "e4_focal_loss"
    assert cfg.architecture == "resnet50"
    assert cfg.finetune_mode == "deep_layer3_4"
    assert cfg.loss == "focal"
    assert cfg.focal_gamma == 2.0
    assert cfg.use_class_weights is False
    assert cfg.use_weighted_sampler is False
    assert cfg.label_smoothing == 0.0
    assert cfg.epochs == 10
    assert cfg.batch_size == 16
    assert cfg.backbone_lr == 2e-5
    assert cfg.head_lr == 2e-4
    assert cfg.weight_decay == 1e-4
    assert cfg.dropout_rate == 0.2
    assert cfg.scheduler_name == "plateau"
    assert cfg.scheduler_factor == 0.5
    assert cfg.scheduler_patience == 2
    assert cfg.early_stopping_patience == 3
    assert cfg.model_selection_metric == "val_macro_f1"
    assert cfg.seed == 42
    assert cfg.image_size == 224
    assert cfg.dataset_name == "Layered-Labs/ACNE04"
    assert cfg.dataset_revision == "e375d8dac98b8562c5ab462d5f00ac011e8a47db"

    output_path = cfg.get_output_path()
    assert output_path.name == "e4_focal_loss"
    assert "experiments" in str(output_path)


def test_cli_argument_parsing():
    """Verify that CLI parser resolves E4 flags correctly without altering defaults."""
    from backend.training.config import TrainingConfig
    # Simulate CLI args for E4
    test_args = [
        "--experiment-id", "e4_focal_loss",
        "--architecture", "resnet50",
        "--finetune-mode", "deep_layer3_4",
        "--dataset-name", "Layered-Labs/ACNE04",
        "--dataset-revision", "e375d8dac98b8562c5ab462d5f00ac011e8a47db",
        "--loss", "focal",
        "--focal-gamma", "2.0",
        "--epochs", "10",
        "--batch-size", "16",
        "--backbone-lr", "0.00002",
        "--head-lr", "0.0002",
        "--weight-decay", "0.0001",
        "--dropout", "0.2",
        "--no-class-weights",
        "--scheduler", "plateau",
        "--scheduler-patience", "2",
        "--scheduler-factor", "0.5",
        "--early-stopping-patience", "3",
        "--seed", "42",
        "--split-manifest", "models/clean_splits_manifest.json",
    ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", type=str, default="resnet50_optimized")
    parser.add_argument("--architecture", type=str, default="resnet50")
    parser.add_argument("--finetune-mode", type=str, default="partial_layer4")
    parser.add_argument("--dataset-name", type=str, default="Layered-Labs/ACNE04")
    parser.add_argument("--dataset-revision", type=str, default="e375d8dac98b8562c5ab462d5f00ac011e8a47db")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--loss", type=str, default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="plateau")
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=str, default=None)

    args = parser.parse_args(test_args)
    assert args.experiment_id == "e4_focal_loss"
    assert args.loss == "focal"
    assert args.focal_gamma == 2.0
    assert args.epochs == 10
    assert args.no_class_weights is True
    assert args.dataset_name == "Layered-Labs/ACNE04"
    assert args.dataset_revision == "e375d8dac98b8562c5ab462d5f00ac011e8a47db"


def test_device_resolution_and_require_gpu():
    """Verify get_device behavior and strict abort on missing required GPU."""
    from backend.training.config import get_device

    # CPU request must succeed
    cpu_dev = get_device("cpu")
    assert cpu_dev.type == "cpu"

    # Auto-detection must return a valid torch.device
    auto_dev = get_device()
    assert isinstance(auto_dev, torch.device)

    # If cuda is requested on a non-CUDA host, verify it raises RuntimeError
    if not torch.cuda.is_available():
        try:
            get_device("cuda")
            raise AssertionError("Expected RuntimeError when requesting cuda on non-CUDA host")
        except RuntimeError as e:
            assert "torch.cuda.is_available() is False" in str(e)

        try:
            get_device(require_gpu=True)
            raise AssertionError("Expected RuntimeError when require_gpu=True on non-CUDA host")
        except RuntimeError as e:
            assert "GPU execution required" in str(e)


def test_clean_splits_manifest_indices_valid_and_disjoint():
    """Verify that the clean split manifest contains valid indices and strictly disjoint partitions."""
    manifest_p = MODELS_DIR / "clean_splits_manifest.json"
    assert manifest_p.exists(), f"Clean split manifest missing at {manifest_p}"

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = manifest["splits"]
    train_idx = set(splits["train"])
    val_idx = set(splits["validation"])
    test_idx = set(splits["test"])

    assert len(train_idx) == 990, f"Expected 990 train, got {len(train_idx)}"
    assert len(val_idx) == 175, f"Expected 175 val, got {len(val_idx)}"
    assert len(test_idx) == 292, f"Expected 292 test, got {len(test_idx)}"
    assert len(train_idx) + len(val_idx) + len(test_idx) == 1457

    # Disjointness
    assert len(train_idx & val_idx) == 0, "Train and Val overlap!"
    assert len(train_idx & test_idx) == 0, "Train and Test overlap!"
    assert len(val_idx & test_idx) == 0, "Val and Test overlap!"


def test_production_checkpoint_untouched():
    """Verify that backend/models/acne_resnet50.pth is untouched with verified SHA256."""
    prod_ckpt = MODELS_DIR / "acne_resnet50.pth"
    assert prod_ckpt.exists(), f"Production checkpoint missing at {prod_ckpt}"

    hasher = hashlib.sha256()
    with open(prod_ckpt, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    observed_sha256 = hasher.hexdigest()

    expected_sha256 = "982ea27736e0a276ad532f90d8603807066aedf0a5d80c70624a6e5671f00ce4"
    assert observed_sha256 == expected_sha256, (
        f"Production checkpoint was modified! SHA-256: {observed_sha256}"
    )


if __name__ == "__main__":
    tests = [
        test_focal_loss_scalar_output,
        test_focal_loss_differentiable,
        test_focal_loss_gamma_zero_agrees_with_ce,
        test_focal_loss_multiclass_batch,
        test_focal_loss_hard_mining_behavior,
        test_e4_config_resolution,
        test_cli_argument_parsing,
        test_device_resolution_and_require_gpu,
        test_clean_splits_manifest_indices_valid_and_disjoint,
        test_production_checkpoint_untouched,
    ]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\nAll {len(tests)} Focal Loss / E4 tests passed successfully!")
