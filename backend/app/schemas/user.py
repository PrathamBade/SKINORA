"""SKINORA Backend — User schemas (Pydantic request/response models)."""

from datetime import datetime

from pydantic import BaseModel, EmailStr


class UserResponse(BaseModel):
    """Public representation of a user (never exposes password hash)."""

    id: str
    email: EmailStr
    username: str
    full_name: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserUpdateRequest(BaseModel):
    """Payload for updating user profile fields."""

    full_name: str | None = None
