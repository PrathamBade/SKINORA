"""ResNet50 inference and evaluation reporting for Skinora's ACNE04 acne-severity model."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import models, transforms

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "acne_resnet50.pth"
METRICS_PATH = ROOT_DIR / "models" / "training_metrics.json"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"

CLASS_NAMES = ["mild", "moderate", "severe", "very_severe"]
IMAGE_SIZE = 224

# Preprocessing identical to training: resize to 256, center-crop to 224, ImageNet normalisation.
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_MODEL_CACHE: Optional[Tuple[nn.Module, List[str], Dict[str, Any]]] = None

# Safe experiment-id pattern: alphanumerics, underscores, hyphens only.
_EXP_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_experiment_path(experiment_id: str) -> Optional[Path]:
    """
    Validate experiment_id and return the resolved experiment directory.
    Returns None if the id is invalid or the resolved path escapes EXPERIMENTS_DIR.
    """
    if not _EXP_ID_RE.match(experiment_id):
        return None
    candidate = (EXPERIMENTS_DIR / experiment_id).resolve()
    try:
        candidate.relative_to(EXPERIMENTS_DIR.resolve())
    except ValueError:
        return None
    return candidate


def _load_model() -> Optional[Tuple[nn.Module, List[str], Dict[str, Any]]]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not MODEL_PATH.exists():
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        classes = checkpoint.get("classes", CLASS_NAMES)
        arch = checkpoint.get("architecture", "resnet50").lower()
        state_dict = checkpoint["model_state_dict"]

        if arch == "resnet50":
            net = models.resnet50(weights=None)
            in_features = net.fc.in_features
            has_sequential_head = any("fc.1." in k for k in state_dict.keys())
            if has_sequential_head:
                dropout_rate = checkpoint.get("dropout_rate", 0.2)
                net.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, len(classes)),
                )
            else:
                net.fc = nn.Linear(in_features, len(classes))
        elif arch == "efficientnet_b0":
            net = models.efficientnet_b0(weights=None)
            in_features = net.classifier[1].in_features
            dropout_rate = checkpoint.get("dropout_rate", 0.2)
            net.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, len(classes)),
            )
        else:
            net = models.resnet50(weights=None)
            net.fc = nn.Linear(net.fc.in_features, len(classes))

        net.load_state_dict(state_dict)
        net.to(device)
        net.eval()

        metadata = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
        _MODEL_CACHE = (net, classes, metadata)
        return _MODEL_CACHE
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None


def model_status() -> dict[str, Any]:
    """Return model status and verification metadata."""
    if not MODEL_PATH.exists():
        return {
            "available": False,
            "status": "not_trained",
            "model": "ResNet50",
            "dataset": "ACNE04",
            "classes": CLASS_NAMES,
            "message": "Train the model with backend/training/train_acne_model.py before enabling image predictions.",
        }

    loaded = _load_model()
    if loaded is None:
        return {
            "available": False,
            "status": "error_loading",
            "model": "ResNet50",
            "dataset": "ACNE04",
            "classes": CLASS_NAMES,
            "message": "Model checkpoint found but failed to load.",
        }

    _, classes, metadata = loaded
    metrics_summary: Dict[str, Any] = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_summary = json.load(f)
        except Exception:
            pass

    # Resolve test_accuracy from training_metrics.json → test_metrics section first,
    # then fall back to the legacy top-level key.
    test_acc = (
        metrics_summary.get("test_metrics", {}).get("test_accuracy")
        or metrics_summary.get("test_accuracy")
    )
    val_acc = (
        metrics_summary.get("validation_metrics", {}).get("best_validation_accuracy")
        or metrics_summary.get("best_validation_accuracy")
        or metadata.get("validation_accuracy")
    )

    return {
        "available": True,
        "status": "ready",
        "model": metadata.get("architecture", "ResNet50"),
        "dataset": metadata.get("primary_dataset", "ACNE04"),
        "classes": classes,
        "experiment_id": metadata.get("experiment_id", "baseline_resnet50"),
        "validation_accuracy": val_acc,
        "validation_macro_f1": metadata.get("validation_macro_f1"),
        "test_accuracy": test_acc,
        "test_macro_f1": metrics_summary.get("test_metrics", {}).get("macro_f1"),
        "message": "Local acne-severity model checkpoint loaded and ready for inference.",
    }


def get_latest_evaluation(experiment_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve full evaluation report and artifact metadata for dashboard display.

    If *experiment_id* is given it must match the safe pattern and resolve within
    EXPERIMENTS_DIR. Otherwise, the most-recently-modified evaluation file is used.
    """
    if experiment_id is not None:
        exp_dir = _validate_experiment_path(experiment_id)
        if exp_dir is None:
            return {
                "available": False,
                "message": "Invalid experiment identifier.",
            }
        eval_file = exp_dir / "evaluation_test.json"
        if not eval_file.exists():
            return {
                "available": False,
                "message": f"No evaluation report found for experiment '{experiment_id}'.",
            }
    else:
        candidate_files = list(EXPERIMENTS_DIR.glob("*/evaluation_test.json"))
        candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidate_files:
            return {
                "available": False,
                "message": "No evaluation reports found. Run backend/training/evaluate_acne_model.py to evaluate.",
            }
        eval_file = candidate_files[0]
        exp_dir = eval_file.parent

    exp_name = exp_dir.name

    try:
        with open(eval_file, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Training history (optional).
        history: list = []
        history_file = exp_dir / "training_history.json"
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)

        # Experiment manifest (optional, written by train_acne_model.py).
        manifest: Dict[str, Any] = {}
        manifest_file = exp_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        # Prefer baseline_metrics.json inside the experiment dir for validation accuracy
        # and split sizes, because evaluation_test.json only records test-split data.
        validation_accuracy: Optional[float] = None
        split_sizes: Optional[Dict[str, int]] = None

        baseline_file = exp_dir / "baseline_metrics.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    bdata = json.load(f)
                validation_accuracy = bdata.get("best_validation_accuracy")
                raw_splits = bdata.get("split_sizes")
                if isinstance(raw_splits, dict):
                    split_sizes = raw_splits
            except Exception:
                pass

        # Fall back to the global training_metrics.json when the experiment matches.
        if validation_accuracy is None and METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                if mdata.get("experiment_id") == exp_name:
                    validation_accuracy = (
                        mdata.get("validation_metrics", {}).get("best_validation_accuracy")
                        or mdata.get("best_validation_accuracy")
                    )
                    if split_sizes is None:
                        ds = mdata.get("dataset_split_sizes")
                        if isinstance(ds, dict):
                            split_sizes = ds
            except Exception:
                pass

        # Normalise split_sizes keys so the frontend always sees train/validation/test.
        if split_sizes:
            split_sizes = {
                "train": split_sizes.get("train") or split_sizes.get("train_images"),
                "validation": split_sizes.get("validation") or split_sizes.get("validation_images"),
                "test": split_sizes.get("test") or split_sizes.get("test_images"),
            }

        has_cm_plot = (exp_dir / "confusion_matrix.png").exists()
        has_curves_plot = (exp_dir / "training_curves.png").exists()

        return {
            "available": True,
            "experiment_id": exp_name,
            "model": manifest.get("architecture", report_data.get("model", "ResNet50")),
            "dataset": "ACNE04",
            "report": report_data,
            "history": history,
            "validation_accuracy": validation_accuracy,
            "split_sizes": split_sizes,
            "has_confusion_matrix_plot": has_cm_plot,
            "has_training_curves_plot": has_curves_plot,
            "artifacts_dir": str(exp_dir),
        }
    except Exception as e:
        return {
            "available": False,
            "message": f"Failed to load evaluation data: {str(e)}",
        }


def predict_acne(image_bytes: bytes) -> dict[str, Any]:
    """Perform model inference on raw image bytes and return predicted class and distribution."""
    loaded = _load_model()
    if loaded is None:
        return {
            "available": False,
            "status": "not_trained",
            "message": "No trained ResNet50 checkpoint is installed. The photo was not classified.",
        }

    try:
        from io import BytesIO
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return {
            "available": False,
            "status": "invalid_image",
            "message": "Could not read this image for model inference.",
        }

    net, classes, metadata = loaded
    device = next(net.parameters()).device
    tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = net(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu()

    index = int(torch.argmax(probs).item())
    confidence = float(probs[index].item())

    # Class probabilities from real model output — never fabricated.
    probabilities = {classes[i]: round(float(probs[i].item()), 4) for i in range(len(classes))}

    return {
        "available": True,
        "status": "prediction_available",
        "model": f"{metadata.get('architecture', 'ResNet50')} fine-tuned on ACNE04",
        "class": classes[index],
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 1),
        "class_probabilities": probabilities,
        "validation_accuracy": metadata.get("validation_accuracy"),
        "warning": (
            "Research prototype only. ACNE04 is a small academic-use dataset; "
            "this prediction is not a diagnosis and may not generalize to selfies, "
            "different lighting, or all skin tones. Seek a dermatologist for medical concerns."
        ),
    }
