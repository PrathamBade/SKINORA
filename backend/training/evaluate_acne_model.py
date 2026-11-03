"""Comprehensive model evaluation module for Skinora acne severity models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from torchvision import transforms

try:
    from .config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, get_device
    from .model import load_model_from_checkpoint
except (ImportError, ValueError):
    import sys
    this_dir = Path(__file__).resolve().parent
    for p in [this_dir, this_dir.parent, this_dir.parent.parent]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    try:
        from training.config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, get_device
        from training.model import load_model_from_checkpoint
    except ImportError:
        from config import CLASSES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, MODELS_DIR, get_device
        from model import load_model_from_checkpoint


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_expected_calibration_error(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(predictions[in_bin] == labels[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Render and save publication-quality confusion matrix image."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).astype(float)
        display_cm = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=(row_sums > 0),
        )
    else:
        display_cm = cm
    cmap = plt.cm.Blues if not normalize else plt.cm.YlGnBu

    im = ax.imshow(display_cm, interpolation="nearest", cmap=cmap)
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=9, fontweight="medium")
    ax.set_yticklabels(classes, fontsize=9, fontweight="medium")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Severity Level", fontsize=10, fontweight="semibold", labelpad=8)
    ax.set_ylabel("True Severity Level", fontsize=10, fontweight="semibold", labelpad=8)

    thresh = display_cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            val_str = f"{display_cm[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if display_cm[i, j] > thresh else "black",
                fontsize=10, fontweight="bold",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_training_curves(
    history: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Training and Validation Curves",
) -> None:
    """Plot Loss and Accuracy / Macro F1 curves over epochs."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] for h in history]
    val_f1 = [h.get("val_macro_f1", 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Loss plot
    ax1.plot(epochs, train_loss, "o-", color="#8A5A44", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_loss, "s--", color="#C2847A", label="Validation Loss", linewidth=2)
    ax1.set_title("Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Accuracy / Macro F1 plot
    ax2.plot(epochs, val_acc, "o-", color="#2D5A27", label="Validation Accuracy", linewidth=2)
    if any(val_f1):
        ax2.plot(epochs, val_f1, "^-.", color="#3D7EAA", label="Validation Macro F1", linewidth=2)
    ax2.set_title("Validation Performance", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.legend(loc="lower right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def evaluate_model_on_split(
    model: nn.Module,
    split_name: str = "test",
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    split_manifest_path: Optional[Path] = None,
    protocol: str = "clean",
) -> Dict[str, Any]:
    """Evaluate model on a designated split and return full academic metrics report."""
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()

    ds = load_dataset("Layered-Labs/ACNE04", split="train")
    split_values = np.asarray(ds["split"])

    manifest_file = split_manifest_path or (MODELS_DIR / "clean_splits_manifest.json")
    used_manifest = False
    is_leakage_free = False

    if protocol.lower() == "clean" and manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        split_key = "validation" if split_name.lower() in {"val", "validation"} else split_name.lower()
        indices = manifest.get("splits", {}).get(split_key, [])
        used_manifest = True
        is_leakage_free = bool(manifest.get("integrity_verification", {}).get("cross_split_leakage_detected") is False)
        print(f"Using clean split manifest from {manifest_file} for split '{split_key}' ({len(indices)} samples, leakage_free={is_leakage_free})")
    elif split_name.lower() in {"val", "validation"}:
        from sklearn.model_selection import train_test_split
        train_indices = [
            i for i, val in enumerate(split_values) if str(val).lower() == "train"
        ]
        train_labels = np.asarray(ds["level"], dtype=np.int64)[train_indices]
        _, val_idx = train_test_split(
            train_indices, test_size=0.15, random_state=42, stratify=train_labels
        )
        indices = val_idx
    else:
        indices = [
            i for i, val in enumerate(split_values)
            if str(val).lower() == split_name.lower()
        ]

    if not indices:
        raise ValueError(f"No samples found for split '{split_name}' in ACNE04 dataset.")

    transform = get_eval_transform()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_probs: List[List[float]] = []
    confidences: List[float] = []

    with torch.inference_mode():
        for idx in indices:
            row = ds[idx]
            img = row["image"].convert("RGB")
            label = int(row["level"])

            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

            pred = int(np.argmax(probs))
            conf = float(probs[pred])

            y_true.append(label)
            y_pred.append(pred)
            y_probs.append(probs.tolist())
            confidences.append(conf)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_probs_arr = np.array(y_probs)
    # Normalize probabilities to sum strictly to 1.0 to avoid scikit-learn log_loss warning
    prob_sums = np.maximum(y_probs_arr.sum(axis=1, keepdims=True), 1e-12)
    y_probs_arr = y_probs_arr / prob_sums
    conf_arr = np.array(confidences)

    # Core metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )

    try:
        ce_loss = float(log_loss(y_true_arr, y_probs_arr, labels=[0, 1, 2, 3]))
    except Exception:
        ce_loss = float("nan")

    ece = compute_expected_calibration_error(conf_arr, y_pred_arr, y_true_arr)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1, 2, 3])

    # Per-class breakdown
    p_cls, r_cls, f_cls, sup_cls = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=[0, 1, 2, 3], average=None, zero_division=0
    )

    class_metrics = {}
    for i, name in enumerate(CLASSES):
        class_metrics[name] = {
            "precision": round(float(p_cls[i]), 4),
            "recall": round(float(r_cls[i]), 4),
            "f1": round(float(f_cls[i]), 4),
            "support": int(sup_cls[i]),
        }

    report = {
        "split": split_name,
        "protocol": protocol,
        "leakage_free": is_leakage_free,
        "split_manifest": str(manifest_file) if used_manifest else None,
        "sample_count": len(indices),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "cross_entropy_loss": round(ce_loss, 4),
        "expected_calibration_error": round(ece, 4),
        "mean_confidence": round(float(np.mean(conf_arr)), 4),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "class_metrics": class_metrics,
    }

    # Save outputs if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON metrics
        suffix = f"_{protocol}" if protocol != "clean" else ""
        metrics_file = output_dir / f"evaluation_{split_name}{suffix}.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # CSV classification report
        csv_file = output_dir / f"classification_report_{split_name}{suffix}.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "precision", "recall", "f1_score", "support"])
            for name, m in class_metrics.items():
                writer.writerow([name, m["precision"], m["recall"], m["f1"], m["support"]])
            writer.writerow([])
            writer.writerow(["macro_avg", report["macro_precision"], report["macro_recall"], report["macro_f1"], len(indices)])
            writer.writerow(["weighted_avg", round(float(weighted_p), 4), round(float(weighted_r), 4), report["weighted_f1"], len(indices)])
            writer.writerow(["accuracy", "", "", report["accuracy"], len(indices)])

        # Raw & Normalized Confusion Matrix plots
        plot_confusion_matrix(
            cm, CLASSES, output_dir / f"confusion_matrix{suffix}.png",
            title=f"ACNE04 {split_name.capitalize()} Set Confusion Matrix ({protocol.capitalize()})",
        )
        plot_confusion_matrix(
            cm, CLASSES, output_dir / f"confusion_matrix_normalized{suffix}.png",
            title=f"ACNE04 {split_name.capitalize()} Set Normalized Confusion Matrix ({protocol.capitalize()})",
            normalize=True,
        )

        # Training history curves if available
        if history:
            plot_training_curves(history, output_dir / "training_curves.png")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate Skinora Acne Model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"], help="Dataset split")
    parser.add_argument("--protocol", type=str, default="clean", choices=["clean", "legacy"], help="Protocol: clean (leakage-free manifest) or legacy (raw ACNE04 split)")
    parser.add_argument("--split-manifest", type=Path, default=None, help="Path to clean_splits_manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save report and plots")
    args = parser.parse_args()

    device = get_device()
    print(f"Loading checkpoint from: {args.checkpoint}")
    model, classes, metadata = load_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Model architecture: {metadata.get('architecture', 'ResNet50')}")

    out = args.output_dir or args.checkpoint.parent
    print(f"Evaluating on ACNE04 '{args.split}' split (protocol={args.protocol})...")
    report = evaluate_model_on_split(
        model,
        split_name=args.split,
        device=device,
        output_dir=out,
        split_manifest_path=args.split_manifest,
        protocol=args.protocol,
    )

    print("\n--- EVALUATION SUMMARY ---")
    print(f"Split:               {report['split']} ({report['sample_count']} images)")
    print(f"Protocol:            {report['protocol']} (leakage_free={report['leakage_free']})")
    print(f"Accuracy:            {report['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy:   {report['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:      {report['macro_f1'] * 100:.2f}%")
    print(f"Cross-Entropy Loss:  {report['cross_entropy_loss']:.4f}")
    print(f"ECE (Calibration):   {report['expected_calibration_error']:.4f}")
    print(f"Artifacts saved to:  {out}")


if __name__ == "__main__":
    main()

