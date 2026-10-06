"""
Loads the PRODUCTION model exactly once at application startup (see main.py's
lifespan handler) and reuses it for every request - this satisfies the
"do not reload the model per request" performance requirement.

Supports both:
  - a deep-learning (Keras) production model -> models/production_model.keras
  - a traditional-ML production model        -> models/production_model.joblib
selected automatically by ml/evaluation/compare_models.py and recorded in
models/production_model_metadata.json.
"""
from __future__ import annotations
import io
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from app.core.config import MODELS_PATH
from ml.config import CLASS_NAMES, IMAGE_SIZE, TRADITIONAL_ML_IMAGE_SIZE
from ml.preprocessing.image_preprocessing import preprocess_pipeline
from ml.features.feature_extraction import extract_features_single

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained production model has been placed under models/.
    This is expected until you actually run the training pipeline on your
    dataset - see README 'Training Models'."""


class InferenceEngine:
    """Singleton-style holder for the in-memory production model."""

    def __init__(self):
        self.model = None
        self.model_type: Optional[str] = None  # "deep_learning" | "traditional"
        self.model_name: Optional[str] = None
        self.scaler = None
        self.pca = None
        self.available_models: dict[str, dict] = {}

    def load(self) -> None:
        meta_path = MODELS_PATH / "production_model_metadata.json"
        if not meta_path.exists():
            logger.warning(
                "No production_model_metadata.json found under %s. "
                "The /api/predict endpoint will return 503 until you train "
                "models (see ml/training/train_all.py) which writes this file.",
                MODELS_PATH,
            )
            return

        meta = json.loads(meta_path.read_text())
        self.model_name = meta["production_model"]
        self.model_type = meta["model_type"]

        if self.model_type == "deep_learning":
            import tensorflow as tf
            artifact_path = MODELS_PATH / meta["artifact"]
            self.model = tf.keras.models.load_model(artifact_path)
        else:
            from joblib import load
            artifact_path = MODELS_PATH / meta["artifact"]
            self.model = load(artifact_path)
            scaler_path = MODELS_PATH / "traditional_feature_scaler.joblib"
            pca_path = MODELS_PATH / "traditional_feature_pca.joblib"
            if scaler_path.exists():
                self.scaler = load(scaler_path)
            if pca_path.exists():
                self.pca = load(pca_path)

        logger.info("Loaded production model '%s' (%s)", self.model_name, self.model_type)

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_ready:
            raise ModelNotAvailableError(
                "No trained production model is available yet. Run the ML "
                "training pipeline (python -m ml.training.train_all) on your "
                "dataset first; it will populate models/production_model.*"
            )

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_rgb = np.array(img)

        if self.model_type == "deep_learning":
            from ml.models.deep_models import preprocess_input_for
            processed = preprocess_pipeline(raw_rgb, IMAGE_SIZE, denoise_enabled=False, enhance_enabled=False)
            processed_uint8 = (processed * 255).astype("uint8")
            backbone_preprocess = preprocess_input_for(self.model_name)
            batch = backbone_preprocess(processed_uint8.astype("float32"))[np.newaxis, ...]
            proba = self.model.predict(batch, verbose=0)[0]
            input_batch = batch
        else:
            processed = preprocess_pipeline(raw_rgb, TRADITIONAL_ML_IMAGE_SIZE, denoise_enabled=True, enhance_enabled=True)
            processed_uint8 = (processed * 255).astype("uint8")
            feats = extract_features_single(processed_uint8).reshape(1, -1)
            if self.scaler is not None:
                feats = self.scaler.transform(feats)
            if self.pca is not None:
                feats = self.pca.transform(feats)
            proba = self.model.predict_proba(feats)[0]
            input_batch = None

        pred_idx = int(np.argmax(proba))
        probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(proba[pred_idx]),
            "probabilities": probabilities,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "raw_image_rgb": raw_rgb,       # kept for Grad-CAM overlay, not serialized
            "predicted_index": pred_idx,
            "input_batch": input_batch,     # preprocessed batch used for prediction (deep models only)
        }


# Module-level singleton, populated once during FastAPI startup lifespan.
engine = InferenceEngine()
