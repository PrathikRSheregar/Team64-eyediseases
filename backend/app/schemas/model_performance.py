from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class ModelPerformanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    model_name: str
    model_version: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: Optional[float] = None
    roc_auc: Optional[float] = None
    inference_time_ms: Optional[float] = None
    training_time_seconds: Optional[float] = None
    n_parameters: Optional[int] = None
    model_size_kb: Optional[float] = None
    is_production: bool
    created_at: datetime
