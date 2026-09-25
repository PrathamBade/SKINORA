"""SKINORA Backend — Authentication routes.

POST /api/v1/auth/register  → create account, return JWT
POST /api/v1/auth/login     → verify credentials, return JWT
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.schemas.auth import AuthResponse, LoginRequest, RegisterRequest, TokenResponse
from app.services import auth_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
async def register(
    data: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Create a new SKINORA account.

    - Email and username must be unique.
    - Password is stored as a bcrypt hash.
    - Returns a JWT access token on success.
    """
    try:
        user = await auth_service.register_user(db, data)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "CONFLICT", "message": str(exc)},
        ) from exc

    token = auth_service.issue_token(user)
    return AuthResponse(
        message="Registration successful.",
        data=TokenResponse(access_token=token),
    )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Log in with email and password",
)
async def login(
    data: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Authenticate with email + password and receive a JWT.

    Returns 401 for invalid credentials (email or password) without
    revealing which field was wrong (prevents email enumeration).
    """
    user = await auth_service.authenticate_user(db, data)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_CREDENTIALS",
                "message": "Invalid email or password.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_service.issue_token(user)
    return AuthResponse(
        message="Login successful.",
        data=TokenResponse(access_token=token),
    )
