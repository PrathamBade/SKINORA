"""
SKINORA ML — Training Script

Trains the ResNet18 acne severity classifier on the ACNE04 dataset.

Usage:
    cd SKINORA/ml
    python -m training.train

Saves:
    ml/models/acne_resnet18.pth       — best model weights (by val accuracy)
    ml/results/training_curves.png    — loss/accuracy plots
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.dataset import get_dataloaders
from training.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MODEL_SAVE_PATH,
    MODELS_DIR,
    NUM_EPOCHS,
    RANDOM_SEED,
    RESULTS_DIR,
    TRAINING_CURVE_IMG,
    WEIGHT_DECAY,
)
from training.model import build_model

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one full pass over the training set.

    Returns: (avg_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # Progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(
                f"    Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True,
            )

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a given DataLoader (no gradients).

    Returns: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def save_training_curves(history: dict, save_path: Path) -> None:
    """Plot and save training + validation loss/accuracy curves."""
    try:
        import matplotlib.pyplot as plt

        epochs = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss
        axes[0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="o")
        axes[0].set_title("Loss per Epoch")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(epochs, history["train_acc"], label="Train Acc", marker="o")
        axes[1].plot(epochs, history["val_acc"], label="Val Acc", marker="o")
        axes[1].set_title("Accuracy per Epoch")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True)

        plt.suptitle("SKINORA — ResNet18 Acne Severity Training", fontsize=13)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Training curves saved: {save_path}")
    except Exception as exc:
        print(f"WARNING: Could not save training curves: {exc}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train() -> None:
    """Full training pipeline: data → model → train → save."""

    print("=" * 65)
    print("SKINORA — Training ResNet18 Acne Severity Classifier")
    print("=" * 65)

    # Setup directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice : {device}")
    print(f"Epochs : {NUM_EPOCHS}")
    print(f"Batch  : {BATCH_SIZE}")
    print(f"LR     : {LEARNING_RATE}")

    # Data
    print("\n--- Loading data ---")
    loaders = get_dataloaders(num_workers=0)

    # Model
    print("\n--- Building model ---")
    model = build_model()
    model = model.to(device)

    # Loss — use class weights to handle imbalance
    # Counts: [491, 623, 177, 115] → inverse-frequency weighting
    train_df_counts = [491, 623, 177, 115]   # approximate from dataset
    total_samples = sum(train_df_counts)
    class_weights = torch.tensor(
        [total_samples / (4 * c) for c in train_df_counts],
        dtype=torch.float32,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimiser — only train the classifier head (fc layer)
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # LR scheduler — reduce LR on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training history
    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    print("\n--- Training ---")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )

        # Step scheduler based on val accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(
            f"  Train  | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} ({train_acc*100:.1f}%)\n"
            f"  Val    | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} ({val_acc*100:.1f}%)\n"
            f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [SAVED] New best model (val_acc={val_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n--- Training complete ---")
    print(f"Best val accuracy: {best_val_acc*100:.2f}% at epoch {best_epoch}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Model saved: {MODEL_SAVE_PATH}")

    # Save history JSON
    history_path = RESULTS_DIR / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(
            {
                **history,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "total_epochs": NUM_EPOCHS,
                "device": str(device),
            },
            f,
            indent=2,
        )
    print(f"Training history: {history_path}")

    # Save training curves
    save_training_curves(history, TRAINING_CURVE_IMG)


if __name__ == "__main__":
    train()
