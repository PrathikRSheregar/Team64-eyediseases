from fastapi import APIRouter
from app.ml.inference import engine

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Health check", description="Reports API status and whether a production ML model is loaded and ready to serve predictions.")
def health_check():
    return {
        "status": "ok",
        "model_loaded": engine.is_ready,
        "model_name": engine.model_name,
        "model_type": engine.model_type,
    }
