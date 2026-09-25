"""
SKINORA Backend — ML Inference Wrapper

Bridges ml/inference.py into the FastAPI backend.

Responsibilities:
  - Resolves the project root path so ml/ is importable from backend/
  - Owns the model singleton lifecycle (load once at startup, reuse per request)
  - Provides graceful fallback: if the model file is missing, the backend
    keeps running and analysis records get status=awaiting_model
  - Exposes a clean interface to analysis_service.py

Design notes:
  - The model is loaded synchronously at startup (blocking, intentional).
    ResNet18 loads in ~0.5s and inference per image is <0.3s — fast enough.
  - Thread-safety: PyTorch's inference is read-only after load; GIL protects
    the singleton. For production scale, move to a process pool or GPU server.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path setup — allow importing from ml/ which lives outside the backend pkg
# ---------------------------------------------------------------------------

# backend/app/ml/inference_wrapper.py  →  go up 4 levels to SKINORA/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Inference service singleton
# ---------------------------------------------------------------------------

class BackendInferenceService:
    """Singleton ML inference service for use within FastAPI request handlers.

    Call `BackendInferenceService.initialize(model_path)` once during app
    startup (inside the lifespan context).  Then call
    `BackendInferenceService.predict(image_path)` from anywhere in the app.
    """

    _service = None           # AcneInferenceService instance
    _available: bool = False  # True only when model loaded successfully
    _load_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    @classmethod
    def initialize(cls, model_path: Optional[Path] = None) -> None:
        """Attempt to load the trained model.

        Called once during FastAPI lifespan startup or on-demand.
        Never raises — failures are logged and the app continues without ML.
        """
        if model_path is None:
            from app.core.config import get_settings
            model_path = get_settings().resolved_model_path

        if not model_path.exists():
            msg = f"ML model not found at: {model_path}. Analysis will run without predictions."
            logger.warning(msg)
            cls._load_error = msg
            cls._available = False
            return

        try:
            from ml.inference import AcneInferenceService
            cls._service = AcneInferenceService(model_path=model_path)
            cls._available = True
            cls._load_error = None
            logger.info(
                "ML inference service loaded successfully from: %s", model_path
            )
        except Exception as exc:
            msg = f"Failed to load ML model: {exc}"
            logger.error(msg)
            cls._load_error = msg
            cls._available = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the model is loaded and ready for inference."""
        if not cls._available and cls._service is None:
            cls.initialize()
        return cls._available

    @classmethod
    def predict(cls, image_path: str | Path) -> Optional[dict]:
        """Run acne severity inference on an image file.

        Args:
            image_path: Path to the uploaded image.

        Returns:
            Dict with keys: severity, confidence, class_name, probabilities
            Returns None if model is not available or inference fails.
        """
        if not cls.is_available() or cls._service is None:
            return None

        try:
            result = cls._service.predict(image_path)
            logger.debug(
                "Inference complete: path=%s severity=%s confidence=%.4f",
                image_path, result["severity"], result["confidence"],
            )
            return result
        except Exception as exc:
            logger.error("Inference failed for %s: %s", image_path, exc)
            return None

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return cls._available

    @classmethod
    def status(cls) -> dict:
        """Return a status dict suitable for the health/system endpoint."""
        return {
            "model_loaded": cls._available,
            "model_path": str(
                cls._service.model_path if cls._service else "N/A"
            ),
            "error": cls._load_error,
        }
