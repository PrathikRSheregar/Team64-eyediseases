from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.session import get_db
from app.api.deps import get_current_admin
from app.models.user import User, UserRole
from app.models.prediction import Prediction
from app.repositories import prediction_repo

router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/overview", summary="Admin dashboard summary", description="Total users, total predictions, disease distribution, model usage, and recent activity. Admin only.")
def admin_overview(db: Session = Depends(get_db), _: User = Depends(get_current_admin)):
    total_users = db.query(func.count(User.id)).scalar()
    total_predictions = prediction_repo.count_all(db)
    disease_distribution = prediction_repo.count_by_class(db)

    model_usage_rows = (
        db.query(Prediction.model_name, func.count(Prediction.id))
        .group_by(Prediction.model_name).all()
    )
    model_usage = {r[0]: r[1] for r in model_usage_rows}

    recent = prediction_repo.list_all(db, limit=10)
    recent_activity = [
        {
            "prediction_id": p.id, "predicted_class": p.predicted_class,
            "confidence": p.confidence, "model_name": p.model_name,
            "created_at": p.created_at.isoformat(),
        }
        for p in recent
    ]

    return {
        "total_users": total_users,
        "total_predictions": total_predictions,
        "disease_distribution": disease_distribution,
        "model_usage": model_usage,
        "recent_activity": recent_activity,
    }
