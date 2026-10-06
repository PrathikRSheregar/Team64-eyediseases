from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.repositories import prediction_repo
from app.schemas.prediction import PredictionHistoryItem, PredictionDetail

router = APIRouter(prefix="/predictions", tags=["Prediction History"])


def _to_probabilities_dict(pred) -> dict[str, float]:
    if not pred.probabilities:
        return {}
    p = pred.probabilities
    return {
        "Cataract": p.cataract_probability,
        "Diabetic Retinopathy": p.diabetic_retinopathy_probability,
        "Glaucoma": p.glaucoma_probability,
        "Normal": p.normal_probability,
    }


@router.get("", response_model=list[PredictionHistoryItem], summary="List the current user's prediction history")
def list_predictions(
    disease: str | None = Query(None, description="Filter by predicted class"),
    model_name: str | None = Query(None, description="Filter by model used"),
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return prediction_repo.list_for_user(db, current_user.id, disease, model_name, limit, offset)


@router.get("/{prediction_id}", response_model=PredictionDetail, summary="Get full details for one prediction")
def get_prediction(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    pred = prediction_repo.get_by_id(db, prediction_id)
    if not pred or (pred.user_id != current_user.id and current_user.role.value != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Prediction not found.")

    return PredictionDetail(
        id=pred.id, predicted_class=pred.predicted_class, confidence=pred.confidence,
        model_name=pred.model_name, image_path=pred.image_path, created_at=pred.created_at,
        probabilities=_to_probabilities_dict(pred), gradcam_path=pred.gradcam_path,
    )


@router.get("/{prediction_id}/image", summary="Fetch the original uploaded image for a prediction")
def get_prediction_image(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    pred = prediction_repo.get_by_id(db, prediction_id)
    if not pred or (pred.user_id != current_user.id and current_user.role.value != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    return FileResponse(pred.image_path)


@router.get("/{prediction_id}/gradcam", summary="Fetch the Grad-CAM heatmap overlay for a prediction")
def get_prediction_gradcam(prediction_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    pred = prediction_repo.get_by_id(db, prediction_id)
    if not pred or (pred.user_id != current_user.id and current_user.role.value != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Prediction not found.")
    if not pred.gradcam_path:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No Grad-CAM available for this prediction.")
    return FileResponse(pred.gradcam_path)
