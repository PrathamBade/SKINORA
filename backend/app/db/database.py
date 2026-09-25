"""
SKINORA Backend — Database Layer

Sets up the SQLAlchemy async engine and session factory.

Design notes:
- Uses async SQLAlchemy 2.x with aiosqlite (SQLite in dev).
- Switching to PostgreSQL requires only changing DATABASE_URL.
- All models inherit from `Base`.
- `get_db` is a FastAPI dependency that yields a session per request.
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,          # Log SQL statements in debug mode
    future=True,
)

# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ---------------------------------------------------------------------------
# Declarative base
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    """Shared base class for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


async def get_db() -> AsyncSession:
    """FastAPI dependency: yield a database session, close it after the request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------


async def init_db() -> None:
    """Create all tables defined by ORM models.

    Called once on application startup.
    In production, prefer Alembic migrations over this.
    """
    # Import models here to ensure they are registered with Base.metadata
    from app.models import analysis, observation, user  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
