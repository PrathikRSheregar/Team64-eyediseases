from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.prediction import PredictionResponse
from app.services import prediction_service

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    "",
    response_model=PredictionResponse,
    summary="Predict eye disease from an uploaded image",
    description=(
        "Uploads a single eye/retinal image, runs it through the current "
        "production model, and returns the predicted class, confidence, the "
        "full probability distribution, and a Grad-CAM explanation link when "
        "available (deep-learning models only)."
    ),
)
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await prediction_service.run_prediction(db, current_user.id, file)


@router.post(
    "/batch",
    response_model=list[PredictionResponse],
    summary="Predict eye disease for multiple images at once",
)
async def predict_batch(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    results = []
    for file in files:
        results.append(await prediction_service.run_prediction(db, current_user.id, file))
    return results
