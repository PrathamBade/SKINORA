"""
SKINORA Backend — Analysis Service

Handles image validation, upload, ML inference dispatch,
analysis record creation, and history retrieval.

Phase 3 update: after a successful image upload, this service calls
BackendInferenceService.predict() to run the real ResNet18 acne severity
model.  If the model is not loaded, analysis falls back gracefully to
status=awaiting_model with no fabricated predictions.
"""

import json
import logging
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.ml.inference_wrapper import BackendInferenceService
from app.models.analysis import Analysis, AnalysisStatus
from app.models.observation import Observation
from app.models.user import User

logger = logging.getLogger(__name__)
settings = get_settings()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Human-readable severity descriptions (mirrors ml/training/config.py)
_SEVERITY_DESCRIPTIONS = {
    0: "No significant acne detected. Skin appears clear.",
    1: "Mild acne detected. Small number of comedones or minor breakouts.",
    2: "Moderate acne detected. Multiple inflammatory lesions or papules.",
    3: "Severe acne detected. Extensive inflammation, nodules, or cystic acne.",
}


# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------


def _validate_image_bytes(data: bytes) -> None:
    """Use Pillow to verify that the bytes represent a valid, readable image.

    Raises:
        ValueError: If the data is not a valid image.
    """
    try:
        import io
        img = Image.open(io.BytesIO(data))
        img.verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Uploaded file is not a valid image: {exc}") from exc


# ---------------------------------------------------------------------------
# Core service functions
# ---------------------------------------------------------------------------


async def upload_image(
    db: AsyncSession,
    user: User,
    filename: str,
    content_type: str,
    file_bytes: bytes,
) -> Analysis:
    """Validate an image upload, run ML inference, and persist results.

    Steps:
      1. Validate file size, extension, content-type, and image integrity.
      2. Save to disk with a safe UUID filename.
      3. Create an Analysis DB record.
      4. Run acne severity inference (if model is loaded).
      5. If inference succeeded, create an Observation and mark completed.
      6. If inference unavailable, mark awaiting_model (no fabricated data).

    Args:
        db: Active async DB session.
        user: Authenticated user performing the upload.
        filename: Original client-provided filename (not trusted for storage).
        content_type: MIME type from the upload.
        file_bytes: Raw image bytes.

    Returns:
        Fully populated Analysis ORM object (with observations loaded).

    Raises:
        ValueError: On any validation failure.
    """
    # ------------------------------------------------------------------
    # 1. Validation
    # ------------------------------------------------------------------
    if len(file_bytes) > settings.max_upload_size_bytes:
        raise ValueError(
            f"File exceeds the maximum allowed size of {settings.max_upload_size_mb} MB."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"File extension '{suffix}' is not supported. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ValueError(
            f"Content type '{content_type}' is not supported. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )

    _validate_image_bytes(file_bytes)

    # ------------------------------------------------------------------
    # 2. Persist image to disk
    # ------------------------------------------------------------------
    safe_name = f"{uuid.uuid4()}{suffix}"
    upload_path = settings.upload_path / safe_name
    upload_path.write_bytes(file_bytes)
    logger.info("Image saved: original=%s stored=%s user_id=%s", filename, safe_name, user.id)

    # ------------------------------------------------------------------
    # 3. Create Analysis record (initially awaiting_model)
    # ------------------------------------------------------------------
    analysis = Analysis(
        user_id=user.id,
        original_filename=filename,
        stored_filename=safe_name,
        image_path=str(upload_path),
        status=AnalysisStatus.AWAITING_MODEL,
        status_message="Image received. Running analysis...",
    )
    db.add(analysis)
    await db.flush()  # Populate analysis.id before creating observations

    # ------------------------------------------------------------------
    # 4 & 5. Run ML inference (synchronous — <0.3s per image on CPU)
    # ------------------------------------------------------------------
    if BackendInferenceService.is_available():
        prediction = BackendInferenceService.predict(str(upload_path))

        if prediction is not None:
            # Build a human-readable description including all class probabilities
            prob_str = ", ".join(
                f"L{k}: {v:.2%}" for k, v in prediction["probabilities"].items()
            )
            description = (
                f"{_SEVERITY_DESCRIPTIONS.get(prediction['severity'], '')}"
                f" | Probabilities: [{prob_str}]"
                f" | Model: ResNet18 baseline (Phase 2)."
            )

            observation = Observation(
                analysis_id=analysis.id,
                observation_type="acne",
                value=prediction["class_name"],
                confidence=prediction["confidence"],
                description=description,
            )
            db.add(observation)

            analysis.status = AnalysisStatus.COMPLETED
            analysis.status_message = (
                f"Analysis complete. Predicted acne severity: "
                f"{prediction['class_name']} "
                f"(confidence {prediction['confidence']:.1%})."
            )

            logger.info(
                "Inference complete: analysis_id=%s severity=%s confidence=%.4f user_id=%s",
                analysis.id, prediction["severity"], prediction["confidence"], user.id,
            )
        else:
            # Inference call failed unexpectedly (logged inside wrapper)
            analysis.status = AnalysisStatus.FAILED
            analysis.status_message = (
                "Image was uploaded but inference failed. Please try again."
            )
    else:
        # Model not loaded — honest placeholder, no fabricated values
        analysis.status = AnalysisStatus.AWAITING_MODEL
        analysis.status_message = (
            "Image uploaded successfully. "
            "ML model is not currently loaded. "
            "No predictions are available."
        )
        logger.info(
            "Analysis created without inference (model not loaded): id=%s", analysis.id
        )

    await db.flush()
    logger.info("Analysis finalised: id=%s status=%s", analysis.id, analysis.status)

    # Re-fetch with selectinload so observations are eagerly loaded for Pydantic serialization
    loaded = await get_analysis_by_id(db, analysis.id, user.id)
    return loaded or analysis


# ---------------------------------------------------------------------------
# History / retrieval
# ---------------------------------------------------------------------------


async def get_analysis_by_id(
    db: AsyncSession, analysis_id: str, user_id: str
) -> Analysis | None:
    """Return an Analysis by ID with observations eagerly loaded."""
    result = await db.execute(
        select(Analysis)
        .options(selectinload(Analysis.observations))
        .where(
            Analysis.id == analysis_id,
            Analysis.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def get_user_analyses(
    db: AsyncSession, user_id: str, limit: int = 20, offset: int = 0
) -> tuple[list[Analysis], int]:
    """Return a paginated list of analyses for a user with observations eagerly loaded."""
    count_result = await db.execute(
        select(func.count(Analysis.id)).where(Analysis.user_id == user_id)
    )
    total = count_result.scalar_one()

    result = await db.execute(
        select(Analysis)
        .options(selectinload(Analysis.observations))
        .where(Analysis.user_id == user_id)
        .order_by(Analysis.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all()), total
