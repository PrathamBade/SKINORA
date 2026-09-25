"""
SKINORA Backend — Recommendations route.

GET /api/v1/recommendations

Returns rule-based skincare guidance, optionally filtered by acne severity.

IMPORTANT: These are general educational guidelines, NOT medical advice.
Personalised ML-driven recommendations will be added in Phase 3.
"""

from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_current_user
from app.models.user import User
from app.services.recommendation_service import get_recommendations

router = APIRouter(prefix="/recommendations", tags=["Recommendations"])


@router.get(
    "",
    summary="Get skincare recommendations",
    description=(
        "Returns general skincare guidance optionally matched to an acne severity level (0–3). "
        "These are educational guidelines only — not personalised medical advice. "
        "Phase 3 will connect this endpoint to real ML observations."
    ),
)
async def recommendations(
    current_user: Annotated[User, Depends(get_current_user)],
    acne_severity: Optional[int] = Query(
        None,
        ge=0,
        le=3,
        description="Acne severity level (0=clear, 1=mild, 2=moderate, 3=severe)",
    ),
) -> dict:
    """Return rule-based skincare recommendations.

    Pass `acne_severity` to get level-specific guidance.
    Omit it for general skincare tips.
    """
    return get_recommendations(acne_severity=acne_severity)
