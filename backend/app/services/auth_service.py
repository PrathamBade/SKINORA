"""
SKINORA Backend — Authentication Service

Handles user registration, login, and current-user lookup.
All password operations use bcrypt via app.core.security.
"""

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, RegisterRequest

logger = logging.getLogger(__name__)


async def register_user(db: AsyncSession, data: RegisterRequest) -> User:
    """Create and persist a new user.

    Raises:
        ValueError: If the email or username is already taken.
    """
    # Check email uniqueness
    result = await db.execute(select(User).where(User.email == data.email))
    if result.scalar_one_or_none():
        raise ValueError("A user with this email already exists.")

    # Check username uniqueness
    result = await db.execute(select(User).where(User.username == data.username))
    if result.scalar_one_or_none():
        raise ValueError("This username is already taken.")

    user = User(
        email=data.email,
        username=data.username,
        hashed_password=hash_password(data.password),
        full_name=data.full_name,
    )
    db.add(user)
    await db.flush()  # Populate id before commit

    logger.info("New user registered: id=%s email=%s", user.id, user.email)
    return user


async def authenticate_user(db: AsyncSession, data: LoginRequest) -> Optional[User]:
    """Verify credentials and return the user, or None on failure.

    Deliberately returns None for both "no such user" and "wrong password"
    to prevent email enumeration attacks.
    """
    result = await db.execute(select(User).where(User.email == data.email))
    user: Optional[User] = result.scalar_one_or_none()

    if user is None or not verify_password(data.password, user.hashed_password):
        logger.warning("Failed login attempt for email=%s", data.email)
        return None

    if not user.is_active:
        logger.warning("Inactive user attempted login: id=%s", user.id)
        return None

    logger.info("User logged in: id=%s", user.id)
    return user


async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Return a User by primary key, or None."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


def issue_token(user: User) -> str:
    """Create a JWT access token for the given user."""
    return create_access_token(subject=user.id)
