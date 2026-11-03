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


def _extract_split_sizes(exp_dir: Path, manifest: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """
    Extract and validate split sizes from experiment artifacts or clean splits manifest.
    Guarantees that train, validation, and test sum to 1457 (ACNE04 total).
    """
    candidates: List[Dict[str, Any]] = []

    # 1. Experiment-local splits_manifest.json
    local_splits = exp_dir / "splits_manifest.json"
    if local_splits.exists():
        try:
            with open(local_splits, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    candidates.append(data)
        except Exception:
            pass

    # 2. Manifest split_counts or split_manifest_path
    if isinstance(manifest, dict):
        if isinstance(manifest.get("split_counts"), dict):
            candidates.append({"split_counts": manifest["split_counts"]})
        if manifest.get("split_manifest_path"):
            ref_path = Path(manifest["split_manifest_path"])
            if not ref_path.is_absolute():
                ref_path = (ROOT_DIR / ref_path).resolve()
            if ref_path.exists():
                try:
                    with open(ref_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict):
                            candidates.append(data)
                except Exception:
                    pass

    # 3. Global clean splits manifest fallback
    clean_manifest = ROOT_DIR / "models" / "clean_splits_manifest.json"
    if clean_manifest.exists():
        try:
            with open(clean_manifest, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    candidates.append(data)
        except Exception:
            pass

    # 4. Baseline metrics file fallback
    baseline_file = exp_dir / "baseline_metrics.json"
    if baseline_file.exists():
        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                bdata = json.load(f)
            if isinstance(bdata.get("split_sizes"), dict):
                candidates.append(bdata["split_sizes"])
        except Exception:
            pass

    # Parse candidate schemas
    for c in candidates:
        if not isinstance(c, dict):
            continue

        train_c: Optional[int] = None
        val_c: Optional[int] = None
        test_c: Optional[int] = None

        # Schema A: direct train_indices, val_indices, test_indices lists
        if isinstance(c.get("train_indices"), list):
            train_c = len(c["train_indices"])
        if isinstance(c.get("val_indices"), list):
            val_c = len(c["val_indices"])
        elif isinstance(c.get("validation_indices"), list):
            val_c = len(c["validation_indices"])
        if isinstance(c.get("test_indices"), list):
            test_c = len(c["test_indices"])

        # Schema B: nested splits dict: { "train": [...], "validation": [...], "test": [...] }
        if (train_c is None or val_c is None or test_c is None) and isinstance(c.get("splits"), dict):
            sp = c["splits"]
            if isinstance(sp.get("train"), list):
                train_c = len(sp["train"])
            if isinstance(sp.get("validation"), list):
                val_c = len(sp["validation"])
            elif isinstance(sp.get("val"), list):
                val_c = len(sp["val"])
            if isinstance(sp.get("test"), list):
                test_c = len(sp["test"])

        # Schema C: nested split_counts dict: { "train": 990, "validation": 175, "test": 292 }
        if (train_c is None or val_c is None or test_c is None) and isinstance(c.get("split_counts"), dict):
            sc = c["split_counts"]
            train_c = sc.get("train")
            val_c = sc.get("validation") if sc.get("validation") is not None else sc.get("val")
            test_c = sc.get("test")

        # Schema D: flat split_sizes dict: { "train": 990, "validation": 175, "test": 292 }
        if train_c is None or val_c is None or test_c is None:
            if "train" in c or "train_images" in c:
                train_c = c.get("train") if c.get("train") is not None else c.get("train_images")
                val_c = c.get("validation") if c.get("validation") is not None else c.get("validation_images")
                if val_c is None:
                    val_c = c.get("val")
                test_c = c.get("test") if c.get("test") is not None else c.get("test_images")

        # Schema E: class_distributions: { "train": [...], "validation": [...], "test": [...] }
        if (train_c is None or val_c is None or test_c is None) and isinstance(c.get("class_distributions"), dict):
            cd = c["class_distributions"]
            if isinstance(cd.get("train"), list):
                train_c = sum(cd["train"])
            if isinstance(cd.get("validation"), list):
                val_c = sum(cd["validation"])
            elif isinstance(cd.get("val"), list):
                val_c = sum(cd["val"])
            if isinstance(cd.get("test"), list):
                test_c = sum(cd["test"])

        # Validate that all 3 counts are non-negative integers summing to 1457
        if (
            isinstance(train_c, int) and isinstance(val_c, int) and isinstance(test_c, int)
            and train_c >= 0 and val_c >= 0 and test_c >= 0
        ):
            total_c = train_c + val_c + test_c
            if total_c == 1457:
                return {
                    "train": train_c,
                    "validation": val_c,
                    "test": test_c,
                    "total": total_c,
                }

    return None


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
            val_candidate = exp_dir / "evaluation_val.json"
            if val_candidate.exists():
                eval_file = val_candidate
            else:
                return {
                    "available": False,
                    "message": f"No evaluation report found for experiment '{experiment_id}'.",
                }
    else:
        candidate_files = list(EXPERIMENTS_DIR.glob("*/evaluation_test.json"))
        candidate_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidate_files:
            val_candidates = list(EXPERIMENTS_DIR.glob("*/evaluation_val.json"))
            val_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            if not val_candidates:
                return {
                    "available": False,
                    "message": "No evaluation reports found. Run backend/training/evaluate_acne_model.py to evaluate.",
                }
            eval_file = val_candidates[0]
        else:
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
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []

        # Experiment manifest (optional, written by train_acne_model.py).
        manifest: Dict[str, Any] = {}
        manifest_file = exp_dir / "manifest.json"
        if manifest_file.exists():
            try:
                with open(manifest_file, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
            except Exception:
                manifest = {}

        # Validation report (optional, recorded validation evaluation for selected checkpoint).
        val_report: Dict[str, Any] = {}
        val_file = exp_dir / "evaluation_val.json"
        if val_file.exists():
            try:
                with open(val_file, "r", encoding="utf-8") as f:
                    val_report = json.load(f)
            except Exception:
                val_report = {}

        # Resolve validation_accuracy and best_epoch:
        validation_accuracy: Optional[float] = None
        best_epoch: Optional[int] = None
        best_val_macro_f1: Optional[float] = None

        # 1. Prefer evaluation_val.json metrics from the selected best checkpoint
        if isinstance(val_report, dict):
            raw_acc = val_report.get("accuracy")
            if raw_acc is not None and isinstance(raw_acc, (int, float)):
                validation_accuracy = float(raw_acc)
            raw_f1 = val_report.get("macro_f1")
            if raw_f1 is not None and isinstance(raw_f1, (int, float)):
                best_val_macro_f1 = float(raw_f1)

        # 2. Prefer manifest.json recorded best metrics and best epoch
        if isinstance(manifest, dict):
            if isinstance(manifest.get("best_epoch"), int):
                best_epoch = manifest["best_epoch"]
            if validation_accuracy is None:
                raw_acc = manifest.get("best_validation_accuracy")
                if raw_acc is None and isinstance(manifest.get("evaluation_metrics"), dict):
                    raw_acc = manifest["evaluation_metrics"].get("accuracy")
                if raw_acc is not None and isinstance(raw_acc, (int, float)):
                    validation_accuracy = float(raw_acc)
            if best_val_macro_f1 is None:
                raw_f1 = manifest.get("best_validation_macro_f1") or manifest.get("best_validation_score")
                if raw_f1 is not None and isinstance(raw_f1, (int, float)):
                    best_val_macro_f1 = float(raw_f1)

        # 3. Match from training_history.json (do NOT confuse final epoch with best checkpoint)
        if isinstance(history, list) and history:
            if best_epoch is not None and validation_accuracy is None:
                for entry in history:
                    if isinstance(entry, dict) and entry.get("epoch") == best_epoch:
                        raw_acc = entry.get("val_accuracy")
                        if raw_acc is not None and isinstance(raw_acc, (int, float)):
                            validation_accuracy = float(raw_acc)
                        if best_val_macro_f1 is None:
                            raw_f1 = entry.get("val_macro_f1")
                            if raw_f1 is not None and isinstance(raw_f1, (int, float)):
                                best_val_macro_f1 = float(raw_f1)
                        break

            if validation_accuracy is None:
                valid_entries = [e for e in history if isinstance(e, dict) and "val_macro_f1" in e]
                if valid_entries:
                    best_entry = max(valid_entries, key=lambda x: x.get("val_macro_f1", -1.0))
                    best_epoch = best_entry.get("epoch")
                    raw_acc = best_entry.get("val_accuracy")
                    if raw_acc is not None and isinstance(raw_acc, (int, float)):
                        validation_accuracy = float(raw_acc)
                    if best_val_macro_f1 is None:
                        raw_f1 = best_entry.get("val_macro_f1")
                        if raw_f1 is not None and isinstance(raw_f1, (int, float)):
                            best_val_macro_f1 = float(raw_f1)

        # 4. Fallback to baseline_metrics.json (e.g. baseline_resnet50 experiment)
        baseline_file = exp_dir / "baseline_metrics.json"
        if validation_accuracy is None and baseline_file.exists():
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    bdata = json.load(f)
                raw_acc = bdata.get("best_validation_accuracy")
                if raw_acc is not None and isinstance(raw_acc, (int, float)):
                    validation_accuracy = float(raw_acc)
            except Exception:
                pass

        # 5. Fallback to global training_metrics.json when experiment matches
        if validation_accuracy is None and METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                if mdata.get("experiment_id") == exp_name:
                    raw_acc = (
                        mdata.get("validation_metrics", {}).get("best_validation_accuracy")
                        or mdata.get("best_validation_accuracy")
                    )
                    if raw_acc is not None and isinstance(raw_acc, (int, float)):
                        validation_accuracy = float(raw_acc)
            except Exception:
                pass

        # Resolve dataset split counts
        split_sizes = _extract_split_sizes(exp_dir, manifest)

        has_cm_plot = (exp_dir / "confusion_matrix.png").exists()
        has_curves_plot = (exp_dir / "training_curves.png").exists()

        return {
            "available": True,
            "experiment_id": exp_name,
            "model": manifest.get("architecture", report_data.get("model", "ResNet50")),
            "dataset": "ACNE04",
            "report": report_data,
            "validation_report": val_report if val_report else None,
            "manifest": manifest if manifest else None,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
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
