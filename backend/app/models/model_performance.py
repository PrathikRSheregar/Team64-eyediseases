import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Float, Integer
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelPerformance(Base):
    __tablename__ = "model_performance"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), default="v1")
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False)
    specificity: Mapped[float] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=True)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=True)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    n_parameters: Mapped[int] = mapped_column(Integer, nullable=True)
    model_size_kb: Mapped[float] = mapped_column(Float, nullable=True)
    is_production: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
