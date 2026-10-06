"""
Orchestrates a single prediction request end to end:
validate -> preprocess -> predict -> Grad-CAM (best effort) -> persist -> respond.
"""
from __future__ import annotations
import logging
import uuid

import numpy as np
from fastapi import HTTPException, UploadFile, status
from sqlalchemy.orm import Session
from PIL import Image

from app.core.config import settings
from app.ml.inference import engine, ModelNotAvailableError
from app.repositories import prediction_repo
from app.services.storage_service import build_storage_key, save_bytes, build_gradcam_key, resolve_path
from app.schemas.prediction import PredictionResponse

logger = logging.getLogger(__name__)


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    if file.content_type not in settings.allowed_image_types_list:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: {settings.allowed_image_types_list}",
        )

    data = await file.read()
    if len(data) > settings.max_upload_size_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.MAX_UPLOAD_SIZE_MB}MB limit.",
        )

    try:
        img = Image.open(__import__("io").BytesIO(data))
        img.verify()
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Uploaded file is not a valid image.")

    return data


def _try_gradcam(result: dict, user_id: str, prediction_id: str) -> str | None:
    """Best-effort Grad-CAM generation - only applies to deep-learning models,
    and any failure here must never break the core prediction response."""
    if result["model_type"] != "deep_learning":
        return None
    try:
        from app.ml.gradcam import make_gradcam_heatmap, overlay_heatmap
        from ml.models.deep_models import last_conv_layer_name

        heatmap = make_gradcam_heatmap(
            img_array=result["input_batch"],
            model=engine.model,
            backbone_layer_index=1,
            last_conv_layer_name=last_conv_layer_name(result["model_name"]),
            pred_index=result["predicted_index"],
        )
        overlay = overlay_heatmap(result["raw_image_rgb"], heatmap)
        overlay_img = Image.fromarray(overlay)
        buf = __import__("io").BytesIO()
        overlay_img.save(buf, format="JPEG", quality=90)

        key = build_gradcam_key(user_id, prediction_id)
        path = save_bytes(key, buf.getvalue())
        return path
    except Exception as exc:  # pragma: no cover - defensive, explainability is best-effort
        logger.warning("Grad-CAM generation failed (prediction still returned): %s", exc)
        return None


async def run_prediction(db: Session, user_id: str, file: UploadFile) -> PredictionResponse:
    image_bytes = await _read_and_validate_upload(file)

    try:
        result = engine.predict(image_bytes)
    except ModelNotAvailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Prediction failed due to an internal error.")

    prediction_id = str(uuid.uuid4())
    storage_key = build_storage_key(user_id, prediction_id, file.filename or "upload.jpg")
    image_path = save_bytes(storage_key, image_bytes)

    gradcam_path = _try_gradcam(result, user_id, prediction_id)

    pred = prediction_repo.create_prediction(
        db, user_id=user_id, image_path=image_path,
        predicted_class=result["predicted_class"], confidence=result["confidence"],
        model_name=result["model_name"], probabilities=result["probabilities"],
        gradcam_path=gradcam_path,
    )

    return PredictionResponse(
        prediction_id=pred.id,
        prediction=result["predicted_class"],
        confidence=round(result["confidence"], 4),
        probabilities={k: round(v, 4) for k, v in result["probabilities"].items()},
        model=result["model_name"],
        model_version="v1",
        gradcam_available=gradcam_path is not None,
        gradcam_url=f"/api/predictions/{pred.id}/gradcam" if gradcam_path else None,
    )
