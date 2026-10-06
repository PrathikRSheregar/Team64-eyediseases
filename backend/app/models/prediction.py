import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Float, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="v1")
    gradcam_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="predictions")
    probabilities = relationship(
        "PredictionProbability", back_populates="prediction",
        uselist=False, cascade="all, delete-orphan"
    )


class PredictionProbability(Base):
    __tablename__ = "prediction_probabilities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    prediction_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("predictions.id", ondelete="CASCADE"), unique=True, nullable=False
    )
    cataract_probability: Mapped[float] = mapped_column(Float, nullable=False)
    diabetic_retinopathy_probability: Mapped[float] = mapped_column(Float, nullable=False)
    glaucoma_probability: Mapped[float] = mapped_column(Float, nullable=False)
    normal_probability: Mapped[float] = mapped_column(Float, nullable=False)

    prediction = relationship("Prediction", back_populates="probabilities")
