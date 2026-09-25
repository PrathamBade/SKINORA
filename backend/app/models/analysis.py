"""
SKINORA Backend — Analysis ORM Model

Represents a single skin analysis session.

Status lifecycle:
    pending → processing → completed | failed
    pending → awaiting_model  (when ML model not yet available)
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class AnalysisStatus(str, PyEnum):
    """Lifecycle states of a skin analysis."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_MODEL = "awaiting_model"  # ML model not yet available


class Analysis(Base):
    """A skin analysis record tied to a user and an uploaded image."""

    __tablename__ = "analyses"

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Image storage
    original_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=True)
    image_path: Mapped[str] = mapped_column(String(512), nullable=True)

    status: Mapped[str] = mapped_column(
        Enum(AnalysisStatus),
        default=AnalysisStatus.AWAITING_MODEL,
        nullable=False,
    )
    status_message: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    user: Mapped["User"] = relationship("User", back_populates="analyses")  # noqa: F821
    observations: Mapped[list["Observation"]] = relationship(  # noqa: F821
        "Observation", back_populates="analysis", lazy="selectin", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Analysis id={self.id!r} status={self.status!r}>"
