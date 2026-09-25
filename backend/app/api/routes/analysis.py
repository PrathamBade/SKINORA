"""
SKINORA Backend — Analysis routes.

POST /api/v1/analysis/upload        → upload image, create analysis record
GET  /api/v1/analysis/history       → list user's past analyses (paginated)
GET  /api/v1/analysis/{analysis_id} → retrieve a specific analysis
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.analysis import (
    AnalysisListResponse,
    AnalysisResponse,
    ErrorDetail,
    ErrorResponse,
    UploadResponse,
)
from app.schemas.observation import ObservationResponse
from app.services import analysis_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["Analysis"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a skin image for analysis",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
    },
)
async def upload_image(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(..., description="Skin image (JPG or PNG, max 10 MB)"),
) -> UploadResponse:
    """Accept a skin image upload, validate it, and create an analysis record.

    - Validates file type by extension AND content-type header.
    - Validates file is a real readable image using Pillow.
    - Saves with a secure UUID-based filename (original name not trusted).
    - Returns an analysis_id that can be used to track the analysis.

    - Automatically dispatches to the ML inference service if loaded.
    - If model is available, produces observations with predicted acne severity and confidence.
    - If model is unavailable, gracefully sets status to 'awaiting_model'.
    """
    file_bytes = await file.read()

    try:
        analysis = await analysis_service.upload_image(
            db=db,
            user=current_user,
            filename=file.filename or "unknown",
            content_type=file.content_type or "",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        msg = str(exc)
        # Map validation errors to appropriate HTTP status codes
        if "size" in msg.lower():
            code = 413
        elif "content type" in msg.lower() or "extension" in msg.lower():
            code = 415
        else:
            code = 400
        raise HTTPException(
            status_code=code,
            detail={"code": "INVALID_IMAGE", "message": msg},
        ) from exc

    logger.info(
        "Upload complete: analysis_id=%s status=%s user_id=%s",
        analysis.id, analysis.status, current_user.id,
    )
    return UploadResponse(
        analysis_id=analysis.id,
        filename=analysis.stored_filename,
        status=analysis.status,
        message=analysis.status_message or "Image uploaded successfully.",
        observations=[
            ObservationResponse.model_validate(obs) for obs in analysis.observations
        ],
    )


@router.get(
    "/history",
    response_model=AnalysisListResponse,
    summary="List analysis history",
    description="Return a paginated list of all skin analyses for the current user.",
)
async def get_history(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(20, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> AnalysisListResponse:
    """Return the authenticated user's analysis history."""
    analyses, total = await analysis_service.get_user_analyses(
        db, current_user.id, limit=limit, offset=offset
    )
    return AnalysisListResponse(
        total=total,
        analyses=[
            AnalysisResponse(
                analysis_id=a.id,
                status=a.status,
                status_message=a.status_message,
                original_filename=a.original_filename,
                observations=[
                    ObservationResponse.model_validate(obs) for obs in a.observations
                ],
                created_at=a.created_at,
            )
            for a in analyses
        ],
    )


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get a specific analysis",
    responses={404: {"model": ErrorResponse, "description": "Not found"}},
)
async def get_analysis(
    analysis_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AnalysisResponse:
    """Retrieve a specific analysis record by ID.

    Returns 404 if the analysis does not exist or belongs to another user.
    """
    analysis = await analysis_service.get_analysis_by_id(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Analysis not found."},
        )

    return AnalysisResponse(
        analysis_id=analysis.id,
        status=analysis.status,
        status_message=analysis.status_message,
        original_filename=analysis.original_filename,
        observations=[
            ObservationResponse.model_validate(obs) for obs in analysis.observations
        ],
        created_at=analysis.created_at,
    )
