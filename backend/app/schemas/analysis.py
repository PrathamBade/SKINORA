"""SKINORA Backend — Analysis schemas (Pydantic request/response models)."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.schemas.observation import ObservationResponse


class MLPrediction(BaseModel):
    """Structured acne severity prediction from the ML model."""

    severity: int
    confidence: float
    class_name: str
    probabilities: dict[str, float]


class AnalysisResponse(BaseModel):
    """Full analysis record returned to the client."""

    analysis_id: str
    status: str
    status_message: Optional[str]
    original_filename: Optional[str]
    observations: list[ObservationResponse]
    created_at: datetime

    model_config = {"from_attributes": True}


class UploadResponse(BaseModel):
    """Response after a successful image upload + analysis."""

    success: bool = True
    analysis_id: str
    filename: str
    status: str
    message: str


class AnalysisListResponse(BaseModel):
    """Paginated list of analyses for a user."""

    success: bool = True
    total: int
    analyses: list[AnalysisResponse]


class MLStatusResponse(BaseModel):
    """Reports whether the ML model is loaded."""

    model_loaded: bool
    model_path: str
    error: Optional[str]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error envelope used across all endpoints."""

    success: bool = False
    error: ErrorDetail
