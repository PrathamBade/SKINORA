"""Controlled, reproducible fine-tuning and evaluation for Skinora acne severity models."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    from .config import (
        CLASSES,
        CLASS_TO_ID,
        EXPERIMENTS_DIR,
        IMAGE_SIZE,
        IMAGENET_MEAN,
        IMAGENET_STD,
        MODELS_DIR,
        TrainingConfig,
        get_device,
        set_seed,
    )
    from .evaluate_acne_model import evaluate_model_on_split, get_eval_transform, plot_confusion_matrix, plot_training_curves
    from .model import build_model, get_parameter_groups, save_checkpoint
except (ImportError, ValueError):
    import sys
    this_dir = Path(__file__).resolve().parent
    for p in [this_dir, this_dir.parent, this_dir.parent.parent]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from training.config import (
            CLASSES,
            CLASS_TO_ID,
            EXPERIMENTS_DIR,
            IMAGE_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
            MODELS_DIR,
            TrainingConfig,
            get_device,
            set_seed,
        )
        from training.evaluate_acne_model import evaluate_model_on_split, get_eval_transform, plot_confusion_matrix, plot_training_curves
        from training.model import build_model, get_parameter_groups, save_checkpoint
    except ImportError:
        from config import (
            CLASSES,
            CLASS_TO_ID,
            EXPERIMENTS_DIR,
            IMAGE_SIZE,
            IMAGENET_MEAN,
            IMAGENET_STD,
            MODELS_DIR,
            TrainingConfig,
            get_device,
            set_seed,
        )
        from evaluate_acne_model import evaluate_model_on_split, get_eval_transform, plot_confusion_matrix, plot_training_curves
        from model import build_model, get_parameter_groups, save_checkpoint


class ACNE04Dataset(Dataset):
    """Dataset wrapper for ACNE04 in-memory samples with transformations."""

    def __init__(self, hf_dataset, indices: List[int], transform: transforms.Compose):
        self.dataset = hf_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.dataset[self.indices[idx]]
        img = row["image"].convert("RGB")
        label = int(row["level"])
        return self.transform(img), label


class FolderDataset(Dataset):
    """Optional dataset loader for structured extra acne images."""

    def __init__(self, items: List[Tuple[Path, int]], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.items[idx]
        with Image.open(path) as source:
            img = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(img), label


def get_train_transform(config: TrainingConfig) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(
            config.image_size,
            scale=(config.aug_scale_min, config.aug_scale_max),
        ),
        transforms.RandomHorizontalFlip(p=0.5 if config.aug_horizontal_flip else 0.0),
        transforms.RandomRotation(degrees=config.aug_rotation_degrees),
        transforms.ColorJitter(
            brightness=config.aug_color_jitter_brightness,
            contrast=config.aug_color_jitter_contrast,
            saturation=config.aug_color_jitter_saturation,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_extra_images(folder: Optional[Path]) -> List[Tuple[Path, int]]:
    items = []
    if folder is None or not folder.exists():
        return items
    for class_name, label in CLASS_TO_ID.items():
        class_dir = folder / class_name
        if not class_dir.exists():
            continue
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append((p, label))
    if items:
        print(f"Found {len(items)} compatible extra images in {folder}")
    return items


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float]:
    """Calculate validation loss, accuracy, balanced accuracy, and macro F1."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds: List[int] = []
    all_targets: List[int] = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total += bs

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = accuracy_score(all_targets, all_preds)
    bal_acc = balanced_accuracy_score(all_targets, all_preds)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    return avg_loss, float(acc), float(bal_acc), float(macro_f1)


def run_training_experiment(config: TrainingConfig) -> Dict[str, Any]:
    """Execute complete controlled experiment with artifact generation."""
    set_seed(config.seed, config.deterministic)
    device = get_device()
    output_dir = config.get_output_path()

    print(f"\n==================================================")
    print(f"Starting Experiment: {config.experiment_id}")
    print(f"Architecture:        {config.architecture}")
    print(f"Fine-tune Mode:      {config.finetune_mode}")
    print(f"Device:              {device}")
    print(f"Output Directory:    {output_dir}")
    print(f"==================================================\n")

    # 1. Dataset loading & integrity verification
    print("Loading ACNE04 dataset from Hugging Face...")
    ds = load_dataset("Layered-Labs/ACNE04", split="train")

    split_values = np.asarray(ds["split"])
    all_labels = np.asarray(ds["level"], dtype=np.int64)

    train_indices = [i for i, v in enumerate(split_values) if str(v).lower() == "train"]
    test_indices = [i for i, v in enumerate(split_values) if str(v).lower() == "test"]

    print(f"Dataset summary: total={len(ds)}, train_split={len(train_indices)}, test_split={len(test_indices)}")

    # Stratified validation split or leakage-free clean split manifest
    manifest_p = Path(config.split_manifest) if config.split_manifest else (MODELS_DIR / "clean_splits_manifest.json")
    if manifest_p.exists():
        with open(manifest_p, "r", encoding="utf-8") as f:
            clean_man = json.load(f)
        train_base = clean_man["splits"]["train"]
        val_rows = clean_man["splits"]["validation"]
        test_rows = clean_man["splits"]["test"]
        print(f"Loaded leakage-free split manifest from {manifest_p}: train={len(train_base)}, val={len(val_rows)}, test={len(test_rows)}")
    else:
        train_labels = all_labels[train_indices]
        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=config.val_split_ratio,
            random_state=config.seed,
            stratify=train_labels,
        )
        train_base = train_idx.tolist()
        val_rows = val_idx.tolist()
        test_rows = test_indices

    # Report class distributions per split
    train_counts = np.bincount(all_labels[train_base], minlength=4)
    val_counts = np.bincount(all_labels[val_rows], minlength=4)
    test_counts = np.bincount(all_labels[test_rows], minlength=4)

    print(f"Split distributions (0=mild, 1=mod, 2=sev, 3=v_sev):")
    print(f"  Train ({len(train_base)}):      {train_counts}")
    print(f"  Val   ({len(val_rows)}):       {val_counts}")
    print(f"  Test  ({len(test_rows)}):      {test_counts}")

    # Optional extra images
    extra = load_extra_images(Path(config.extra_data_dir) if config.extra_data_dir else None)
    extra_train, extra_val = [], []
    if extra:
        extra_labels = np.asarray([label for _, label in extra])
        e_train, e_val = train_test_split(
            np.arange(len(extra)),
            test_size=config.val_split_ratio,
            random_state=config.seed,
            stratify=extra_labels if len(np.unique(extra_labels)) > 1 else None,
        )
        extra_train = [extra[i] for i in e_train]
        extra_val = [extra[i] for i in e_val]

    # Data loaders
    train_transform = get_train_transform(config)
    eval_transform = get_eval_transform(config.image_size)

    train_ds = torch.utils.data.ConcatDataset([
        ACNE04Dataset(ds, train_base, train_transform),
        *([FolderDataset(extra_train, train_transform)] if extra_train else []),
    ])
    val_ds = torch.utils.data.ConcatDataset([
        ACNE04Dataset(ds, val_rows, eval_transform),
        *([FolderDataset(extra_val, eval_transform)] if extra_val else []),
    ])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 2. Build model with architecture & layer freezing
    model = build_model(config, num_classes=len(CLASSES)).to(device)

    # 3. Class weights calculation for balanced loss
    if config.use_class_weights:
        total_train = len(train_base) + len(extra_train)
        combined_counts = train_counts.copy()
        for _, lbl in extra_train:
            combined_counts[lbl] += 1
        class_weights = total_train / (len(CLASSES) * np.maximum(combined_counts, 1).astype(float))
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float, device=device)
        print(f"Using balanced class weights: {np.round(class_weights, 3)}")
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # 4. Optimizer and scheduler with differential learning rates
    param_groups = get_parameter_groups(model, config)
    optimizer = torch.optim.AdamW(param_groups)

    if config.scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if config.model_selection_metric.endswith("f1") or config.model_selection_metric.endswith("acc") else "min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    elif config.scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    else:
        scheduler = None

    # Mixed precision scaler (CUDA only)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 5. Training loop with Validation Macro F1 Model Selection
    history: List[Dict[str, Any]] = []
    best_selection_score = -1.0
    best_epoch = 0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    best_val_loss = 0.0
    no_improve = 0
    best_checkpoint_path = output_dir / f"{config.experiment_id}_best.pth"

    # Save exact split indices manifest for 100% reproducibility
    splits_manifest = {
        "seed": config.seed,
        "val_split_ratio": config.val_split_ratio,
        "train_indices": train_base,
        "val_indices": val_rows,
        "test_indices": test_rows,
        "class_distributions": {
            "train": train_counts.tolist(),
            "validation": val_counts.tolist(),
            "test": test_counts.tolist(),
        },
    }
    with open(output_dir / "splits_manifest.json", "w", encoding="utf-8") as f:
        json.dump(splits_manifest, f, indent=2)

    print("\nStarting training epochs...")
    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        seen = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            seen += bs

        train_loss = running_loss / max(seen, 1)

        # Validation evaluation
        val_loss, val_acc, val_bal_acc, val_macro_f1 = evaluate_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[-1]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                target_score = val_macro_f1 if config.model_selection_metric == "val_macro_f1" else val_loss
                scheduler.step(target_score)
            else:
                scheduler.step()

        duration = round(time.time() - t0, 1)
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_balanced_accuracy": round(val_bal_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "lr": current_lr,
            "duration_sec": duration,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{config.epochs:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc*100:.1f}% | "
            f"val_macro_f1={val_macro_f1*100:.1f}% | "
            f"lr={current_lr:.2e} | "
            f"{duration}s"
        )

        # Model selection metric check
        if config.model_selection_metric == "val_macro_f1":
            metric_val = val_macro_f1
        elif config.model_selection_metric == "val_acc":
            metric_val = val_acc
        else:
            metric_val = -val_loss

        if metric_val > best_selection_score:
            best_selection_score = metric_val
            best_epoch = epoch
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            no_improve = 0

            save_checkpoint(
                model=model,
                config=config,
                metrics={
                    "val_accuracy": round(val_acc, 4),
                    "val_macro_f1": round(val_macro_f1, 4),
                    "val_loss": round(val_loss, 4),
                },
                filepath=best_checkpoint_path,
            )
            print(f"  --> Saved new best checkpoint at epoch {epoch} ({config.model_selection_metric}={metric_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= config.early_stopping_patience:
                print(f"Early stopping triggered after {no_improve} epochs without improvement.")
                break

    # Save training history JSON
    history_file = output_dir / "training_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # 6. Evaluation using Best Checkpoint (Validation by default, or Test if evaluate_test=True)
    eval_split = "test" if getattr(config, "evaluate_test", False) else "val"
    print(f"\nLoading best checkpoint (from epoch {best_epoch}) for final {eval_split} set evaluation...")
    best_model = build_model(config, num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=True)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model.eval()

    eval_report = evaluate_model_on_split(
        model=best_model,
        split_name=eval_split,
        device=device,
        output_dir=output_dir,
        history=history,
    )

    # 7. Experiment manifest
    manifest = {
        "experiment_id": config.experiment_id,
        "description": config.description,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": config.architecture,
        "finetune_mode": config.finetune_mode,
        "pretrained_weights": config.pretrained_weights,
        "seed": config.seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "model_selection_metric": config.model_selection_metric,
        "best_validation_score": round(best_selection_score, 4),
        "best_validation_accuracy": round(best_val_acc, 4),
        "best_validation_macro_f1": round(best_val_macro_f1, 4),
        "best_validation_loss": round(best_val_loss, 4),
        "split_counts": {
            "train": len(train_base) + len(extra_train),
            "validation": len(val_rows) + len(extra_val),
            "test": len(test_rows),
        },
        "class_distributions": {
            "train": train_counts.tolist(),
            "validation": val_counts.tolist(),
            "test": test_counts.tolist(),
        },
        "evaluation_split": eval_split,
        "evaluation_metrics": {
            "accuracy": eval_report["accuracy"],
            "balanced_accuracy": eval_report["balanced_accuracy"],
            "macro_precision": eval_report["macro_precision"],
            "macro_recall": eval_report["macro_recall"],
            "macro_f1": eval_report["macro_f1"],
            "weighted_f1": eval_report["weighted_f1"],
            "cross_entropy_loss": eval_report["cross_entropy_loss"],
            "expected_calibration_error": eval_report["expected_calibration_error"],
            "per_class": eval_report.get("per_class", {}),
        },
        "checkpoint_path": str(best_checkpoint_path),
    }

    if eval_split == "test":
        manifest["test_metrics"] = manifest["evaluation_metrics"]

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n==================================================")
    print(f"Experiment '{config.experiment_id}' Completed!")
    print(f"Best Epoch:               {best_epoch}")
    print(f"Best Val Macro F1 (Loop): {best_selection_score*100:.2f}%")
    print(f"{eval_split.capitalize()} Accuracy:            {eval_report['accuracy']*100:.2f}%")
    print(f"{eval_split.capitalize()} Balanced Accuracy:   {eval_report['balanced_accuracy']*100:.2f}%")
    print(f"{eval_split.capitalize()} Macro F1:            {eval_report['macro_f1']*100:.2f}%")
    print(f"{eval_split.capitalize()} Loss:                {eval_report['cross_entropy_loss']:.4f}")
    print(f"Artifacts saved in:       {output_dir}")
    print(f"==================================================\n")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Train Skinora Acne Model")
    parser.add_argument("--experiment-id", type=str, default="resnet50_optimized")
    parser.add_argument("--architecture", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--finetune-mode", type=str, default="partial_layer4", choices=["frozen", "partial_layer4", "deep_layer3_4", "full"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-manifest", type=str, default=None, help="Path to clean_splits_manifest.json")
    parser.add_argument("--extra-data-dir", type=str, default=None)
    parser.add_argument("--aug-rotation", type=float, default=12.0, help="Random rotation degrees")
    parser.add_argument("--aug-brightness", type=float, default=0.08, help="Color jitter brightness factor")
    parser.add_argument("--aug-contrast", type=float, default=0.08, help="Color jitter contrast factor")
    parser.add_argument("--aug-saturation", type=float, default=0.05, help="Color jitter saturation factor")
    parser.add_argument("--no-aug-flip", action="store_true", help="Disable random horizontal flip")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate on held-out test split (default: evaluate on validation set only)")
    parser.add_argument("--set-as-production", action="store_true", help="Copy checkpoint and metrics to models/ for backend serving")
    args = parser.parse_args()

    cfg = TrainingConfig(
        experiment_id=args.experiment_id,
        architecture=args.architecture,
        finetune_mode=args.finetune_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        use_class_weights=(not args.no_class_weights),
        label_smoothing=args.label_smoothing,
        scheduler_name=args.scheduler,
        seed=args.seed,
        split_manifest=args.split_manifest,
        extra_data_dir=args.extra_data_dir,
        aug_rotation_degrees=args.aug_rotation,
        aug_color_jitter_brightness=args.aug_brightness,
        aug_color_jitter_contrast=args.aug_contrast,
        aug_color_jitter_saturation=args.aug_saturation,
        aug_horizontal_flip=(not args.no_aug_flip),
        evaluate_test=args.evaluate_test,
    )

    manifest = run_training_experiment(cfg)

    if args.set_as_production:
        ckpt_src = Path(manifest["checkpoint_path"])
        arch = manifest.get("architecture", "resnet50").lower()
        if arch == "resnet50":
            prod_ckpt = MODELS_DIR / "acne_resnet50.pth"
        else:
            prod_ckpt = MODELS_DIR / f"acne_{arch}.pth"
        prod_metrics = MODELS_DIR / "training_metrics.json"

        # Copy checkpoint
        import shutil
        shutil.copy2(ckpt_src, prod_ckpt)

        # Format training_metrics.json for production
        prod_data = {
            "model": manifest["architecture"],
            "experiment_id": manifest["experiment_id"],
            "primary_dataset": "ACNE04",
            "pretraining": manifest["pretrained_weights"],
            "classes": CLASSES,
            "dataset_split_sizes": {
                "train_images": manifest["split_counts"]["train"],
                "validation_images": manifest["split_counts"]["validation"],
                "test_images": manifest["split_counts"]["test"],
            },
            "validation_metrics": {
                "best_validation_accuracy": manifest.get("best_validation_accuracy", manifest.get("best_validation_score")),
                "best_validation_macro_f1": manifest.get("best_validation_macro_f1", manifest.get("best_validation_score")),
            },
            "test_metrics": {
                "test_accuracy": (manifest.get("test_metrics") or {}).get("accuracy"),
                "test_accuracy_percent": round(manifest["test_metrics"]["accuracy"] * 100, 2) if manifest.get("test_metrics") else None,
                "balanced_accuracy": (manifest.get("test_metrics") or {}).get("balanced_accuracy"),
                "macro_precision": (manifest.get("test_metrics") or {}).get("macro_precision"),
                "macro_recall": (manifest.get("test_metrics") or {}).get("macro_recall"),
                "macro_f1": (manifest.get("test_metrics") or {}).get("macro_f1"),
                "weighted_f1": (manifest.get("test_metrics") or {}).get("weighted_f1"),
                "cross_entropy_loss": (manifest.get("test_metrics") or {}).get("cross_entropy_loss"),
                "expected_calibration_error": (manifest.get("test_metrics") or {}).get("expected_calibration_error"),
            },
            "best_validation_accuracy": manifest.get("best_validation_accuracy", manifest.get("best_validation_score")),
            "test_accuracy": (manifest.get("test_metrics") or {}).get("accuracy"),
            "test_loss": (manifest.get("test_metrics") or {}).get("cross_entropy_loss"),
            "checkpoint": str(prod_ckpt),
            "updated_at": manifest["timestamp"],
        }
        with open(prod_metrics, "w", encoding="utf-8") as f:
            json.dump(prod_data, f, indent=2)

        print(f"Updated production model at {prod_ckpt} and metrics at {prod_metrics}")


if __name__ == "__main__":
    main()
