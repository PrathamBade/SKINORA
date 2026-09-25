"""
SKINORA Backend — ML status route.

GET /api/v1/ml/status  → reports whether the inference model is loaded.

Used by frontend, monitoring, and operators to check ML readiness
without having to upload an image to find out.
"""

from fastapi import APIRouter

from app.ml.inference_wrapper import BackendInferenceService
from app.schemas.analysis import MLStatusResponse

router = APIRouter(prefix="/ml", tags=["ML"])


@router.get(
    "/status",
    response_model=MLStatusResponse,
    summary="ML model status",
    description=(
        "Reports whether the acne severity ML model is currently loaded "
        "and ready for inference. If `model_loaded` is false, uploaded images "
        "will be stored but predictions will not be available."
    ),
)
async def ml_status() -> MLStatusResponse:
    """Return current ML inference service status."""
    status = BackendInferenceService.status()
    return MLStatusResponse(**status)
