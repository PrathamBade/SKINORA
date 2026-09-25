"""
SKINORA Backend — Observations route.

GET /api/v1/observations/{analysis_id}

Returns observations for a given analysis.
Observations contain real values only when populated by an ML model.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.schemas.observation import ObservationResponse
from app.services import analysis_service

router = APIRouter(prefix="/observations", tags=["Observations"])


@router.get(
    "/{analysis_id}",
    response_model=list[ObservationResponse],
    summary="Get observations for an analysis",
    description=(
        "Returns all skin observations recorded for a given analysis. "
        "Confidence values are only present when a trained ML model has run."
    ),
)
async def get_observations(
    analysis_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[ObservationResponse]:
    """Return observations for the given analysis (user must own it)."""
    analysis = await analysis_service.get_analysis_by_id(db, analysis_id, current_user.id)
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": "Analysis not found."},
        )

    return [ObservationResponse.model_validate(obs) for obs in analysis.observations]
