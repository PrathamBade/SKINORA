"""Unit tests for metric calculations, ECE, and confusion matrix utilities."""
import tempfile
from pathlib import Path
import numpy as np
import pytest

from Skinora.backend.training.config import CLASSES
from Skinora.backend.training.evaluate_acne_model import (
    compute_expected_calibration_error,
    plot_confusion_matrix,
    plot_training_curves,
)


def test_compute_ece():
    # Perfect calibration test
    confidences = np.array([0.9, 0.8, 0.7, 0.6])
    predictions = np.array([0, 1, 2, 3])
    labels = np.array([0, 1, 2, 3])
    ece = compute_expected_calibration_error(confidences, predictions, labels, num_bins=5)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_plot_confusion_matrix():
    cm = np.array([[10, 2, 0, 0], [1, 15, 2, 0], [0, 3, 8, 1], [0, 0, 1, 5]])
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "cm.png"
        plot_confusion_matrix(cm, CLASSES, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000


def test_plot_training_curves():
    history = [
        {"epoch": 1, "train_loss": 1.2, "val_loss": 1.0, "val_accuracy": 0.55, "val_macro_f1": 0.50},
        {"epoch": 2, "train_loss": 0.9, "val_loss": 0.8, "val_accuracy": 0.65, "val_macro_f1": 0.62},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out_png = Path(tmpdir) / "curves.png"
        plot_training_curves(history, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000
