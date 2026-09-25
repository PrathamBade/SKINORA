"""
SKINORA Backend — Observation ORM Model

Represents a single skin observation within an analysis.
Each observation describes one measurable skin characteristic.

NOTE: Observations with real ML-derived values are only created
when a trained model is available.  Fields like `confidence` are
left NULL until real inference is performed.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Observation(Base):
    """A single skin characteristic observation within an analysis."""

    __tablename__ = "observations"

    # ------------------------------------------------------------------
    # Columns
    # ------------------------------------------------------------------

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    analysis_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Observation type — e.g. "acne", "pigmentation", "redness", "pores"
    observation_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Human-readable value — e.g. "mild", "moderate", "severe"
    value: Mapped[str] = mapped_column(String(255), nullable=True)

    # Numeric confidence score [0.0, 1.0] — NULL when not available
    confidence: Mapped[float] = mapped_column(Float, nullable=True)

    # Optional human-readable description
    description: Mapped[str] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    analysis: Mapped["Analysis"] = relationship(  # noqa: F821
        "Analysis", back_populates="observations"
    )

    def __repr__(self) -> str:
        return (
            f"<Observation id={self.id!r} "
            f"type={self.observation_type!r} value={self.value!r}>"
        )
