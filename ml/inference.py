"""
SKINORA ML — Inference Service

Provides a clean interface for running acne severity inference on a
single image.  This module is imported by the FastAPI backend in Phase 3.

CURRENT STATUS: Awaiting trained model (Phase 2 training).
Once ml/models/acne_resnet18.pth exists, this service is ready to use.

Usage (after training):
    from ml.inference import AcneInferenceService
    service = AcneInferenceService()
    result = service.predict("path/to/image.jpg")
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.training.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
)
from ml.training.model import build_model


class AcneInferenceService:
    """Loads the trained ResNet18 model and runs inference on a single image.

    Designed for integration with the FastAPI backend in Phase 3.

    Attributes:
        model_path: Path to the saved .pth weights file.
        device: PyTorch device (cpu or cuda).
        model: Loaded model in eval mode.
        transform: Preprocessing transform applied to input images.
    """

    _instance: Optional["AcneInferenceService"] = None  # Singleton cache

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_SAVE_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self._load_model()

    def _load_model(self) -> None:
        """Load model weights from disk.

        Raises:
            FileNotFoundError: If the model file does not exist.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at: {self.model_path}\n"
                "Train the model first: cd ml && python -m training.train"
            )

        self.model = build_model()
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls) -> "AcneInferenceService":
        """Return a singleton instance (lazy-loaded).

        Use this in FastAPI lifespan / dependency injection to avoid
        loading the model on every request.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @torch.no_grad()
    def predict(self, image_path: str | Path) -> dict:
        """Run acne severity inference on a single image file.

        Args:
            image_path: Path to a JPEG or PNG image.

        Returns:
            Dict with:
                severity      — predicted class (0–3)
                confidence    — probability of the predicted class [0.0, 1.0]
                class_name    — human-readable label
                probabilities — per-class probabilities {0: ..., 1: ..., ...}

        Raises:
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image cannot be loaded.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not load image: {exc}") from exc

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

        severity = int(probs.argmax().item())
        confidence = float(probs[severity].item())
        probabilities = {i: round(float(probs[i].item()), 4) for i in range(NUM_CLASSES)}

        return {
            "severity": severity,
            "confidence": round(confidence, 4),
            "class_name": CLASS_NAMES[severity],
            "probabilities": probabilities,
        }

    @property
    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self.model is not None


def model_exists() -> bool:
    """Quick check: does the trained model file exist on disk?"""
    return MODEL_SAVE_PATH.exists()
