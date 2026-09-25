"""SKINORA Backend — Observation schemas (Pydantic request/response models)."""

from pydantic import BaseModel


class ObservationResponse(BaseModel):
    """A single skin observation returned to the client."""

    id: str
    observation_type: str
    value: str | None
    confidence: float | None
    description: str | None

    model_config = {"from_attributes": True}
