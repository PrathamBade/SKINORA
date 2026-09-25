"""
SKINORA ML — Model Evaluation

Evaluates the trained model on the held-out test set.

Metrics:
    - Accuracy
    - Per-class Precision, Recall, F1
    - Macro-averaged F1
    - Weighted F1
    - Confusion matrix (saved as PNG)

Usage:
    cd SKINORA/ml
    python -m training.evaluate

Saves:
    ml/results/evaluation_results.json
    ml/results/confusion_matrix.png
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.dataset import get_dataloaders
from training.config import (
    CLASS_NAMES,
    CONFUSION_MATRIX_IMG,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    RESULTS_DIR,
    RESULTS_JSON,
)
from training.model import build_model


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Run model over a DataLoader and collect (all_labels, all_preds)."""
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds


def compute_metrics(
    labels: list[int],
    preds: list[int],
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Compute accuracy, per-class precision/recall/F1, macro+weighted F1."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_recall_fscore_support,
    )

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {}
    for i in range(num_classes):
        per_class[f"class_{i}"] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }

    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "classification_report": report,
    }


def save_confusion_matrix(
    labels: list[int],
    preds: list[int],
    save_path: Path,
    num_classes: int = NUM_CLASSES,
) -> None:
    """Plot and save a labelled confusion matrix as a PNG."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)

        tick_marks = np.arange(num_classes)
        short_names = ["Level 0\n(Clear)", "Level 1\n(Mild)", "Level 2\n(Moderate)", "Level 3\n(Severe)"]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_title("SKINORA — Acne Severity Confusion Matrix\n(Test Set)", fontsize=12)

        plt.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved: {save_path}")
    except Exception as exc:
        print(f"WARNING: Could not save confusion matrix: {exc}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def evaluate() -> dict:
    """Load the best saved model and evaluate on the test set."""
    print("=" * 65)
    print("SKINORA — Model Evaluation on Test Set")
    print("=" * 65)

    if not MODEL_SAVE_PATH.exists():
        print(f"\nERROR: Model not found at {MODEL_SAVE_PATH}")
        print("Run `python -m training.train` first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Model:  {MODEL_SAVE_PATH}")

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model = model.to(device)
    print("Model loaded successfully.")

    # Load test data
    print("\n--- Loading test set ---")
    loaders = get_dataloaders(num_workers=0)

    # Collect predictions
    print("\n--- Running inference ---")
    labels, preds = collect_predictions(model, loaders["test"], device)

    # Compute metrics
    metrics = compute_metrics(labels, preds)

    # Print results
    print("\n--- Results ---")
    print(f"  Test Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"  Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1    : {metrics['weighted_f1']:.4f}")
    print("\nPer-class metrics:")
    for cls, vals in metrics["per_class"].items():
        print(
            f"  {cls:8s} | "
            f"Precision: {vals['precision']:.4f} | "
            f"Recall: {vals['recall']:.4f} | "
            f"F1: {vals['f1']:.4f} | "
            f"Support: {vals['support']}"
        )
    print("\nFull classification report:")
    print(metrics["classification_report"])

    # Save metrics JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_with_meta = {
        **metrics,
        "model": str(MODEL_SAVE_PATH.name),
        "test_size": len(labels),
    }
    # Remove the text report from JSON (it's verbose)
    results_json = {k: v for k, v in results_with_meta.items() if k != "classification_report"}
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved: {RESULTS_JSON}")

    # Save confusion matrix
    save_confusion_matrix(labels, preds, CONFUSION_MATRIX_IMG)

    return results_with_meta


if __name__ == "__main__":
    evaluate()
