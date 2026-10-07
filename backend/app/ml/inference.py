"""
Production inference engine.

Supports:
  - Deep-learning Keras models
  - Traditional ML models
  - Hybrid ResNet + Traditional Features + RBF SVM

The Hybrid production pipeline is:

Image
    -> ResNet Dense(128) features
    + HOG/color/statistical features
    -> feature fusion
    -> StandardScaler
    -> PCA
    -> RBF SVM
    -> prediction
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
from ml.config import (
    CLASS_NAMES,
    IMAGE_SIZE,
    TRADITIONAL_ML_IMAGE_SIZE,
)
from ml.preprocessing.image_preprocessing import preprocess_pipeline
from ml.features.feature_extraction import extract_features_single

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained production model is available."""


class InferenceEngine:
    """Singleton-style holder of the in-memory production model."""

    def __init__(self):
        self.model = None
        self.model_type: Optional[str] = None
        self.model_name: Optional[str] = None

        # Traditional ML
        self.scaler = None
        self.pca = None

        # Hybrid ML
        self.hybrid_feature_extractor = None
        self.hybrid_scaler = None
        self.hybrid_pca = None
        self.hybrid_svm = None

        self.available_models: dict[str, dict] = {}

    def load(self) -> None:
        meta_path = MODELS_PATH / "production_model_metadata.json"

        if not meta_path.exists():
            logger.warning(
                "No production_model_metadata.json found under %s.",
                MODELS_PATH,
            )
            return

        meta = json.loads(meta_path.read_text())

        self.model_name = meta["production_model"]
        self.model_type = meta["model_type"]

        # ---------------------------------------------------------
        # DEEP LEARNING MODEL
        # ---------------------------------------------------------
        if self.model_type == "deep_learning":
            import tensorflow as tf

            artifact_path = MODELS_PATH / meta["artifact"]

            self.model = tf.keras.models.load_model(
                artifact_path
            )

        # ---------------------------------------------------------
        # HYBRID MODEL
        # ---------------------------------------------------------
        elif self.model_type == "hybrid":
            import tensorflow as tf
            from joblib import load
            from tensorflow.keras.models import Model

            # ResNet is used only as a feature extractor.
            resnet_path = MODELS_PATH / "resnet_best.keras"

            logger.info(
                "Loading Hybrid ResNet feature extractor from %s",
                resnet_path,
            )

            resnet = tf.keras.models.load_model(
                resnet_path,
                compile=False,
            )

            # Same layer used during Hybrid training:
            #
            # ResNet
            # -> Dropout
            # -> Dense(128)
            # -> Dropout
            # -> Dense(4)
            #
            # The -3 layer is the learned 128-dimensional representation.
            feature_layer = resnet.layers[-3]

            self.hybrid_feature_extractor = Model(
                inputs=resnet.input,
                outputs=feature_layer.output,
            )

            # Hybrid SVM
            svm_path = (
                MODELS_PATH
                / "hybrid_resnet_traditional_svm.joblib"
            )

            scaler_path = (
                MODELS_PATH
                / "hybrid_feature_scaler.joblib"
            )

            pca_path = (
                MODELS_PATH
                / "hybrid_feature_pca.joblib"
            )

            self.hybrid_svm = load(svm_path)
            self.hybrid_scaler = load(scaler_path)
            self.hybrid_pca = load(pca_path)

            logger.info(
                "Loaded Hybrid ResNet + Traditional Features + RBF SVM"
            )

        # ---------------------------------------------------------
        # TRADITIONAL MODEL
        # ---------------------------------------------------------
        else:
            from joblib import load

            artifact_path = MODELS_PATH / meta["artifact"]

            self.model = load(artifact_path)

            scaler_path = (
                MODELS_PATH
                / "traditional_feature_scaler.joblib"
            )

            pca_path = (
                MODELS_PATH
                / "traditional_feature_pca.joblib"
            )

            if scaler_path.exists():
                self.scaler = load(scaler_path)

            if pca_path.exists():
                self.pca = load(pca_path)

        logger.info(
            "Loaded production model '%s' (%s)",
            self.model_name,
            self.model_type,
        )

    @property
    def is_ready(self) -> bool:
        if self.model_type == "hybrid":
            return (
                self.hybrid_feature_extractor is not None
                and self.hybrid_svm is not None
                and self.hybrid_scaler is not None
                and self.hybrid_pca is not None
            )

        return self.model is not None

    def predict(self, image_bytes: bytes) -> dict:
        if not self.is_ready:
            raise ModelNotAvailableError(
                "No trained production model is available."
            )

        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        raw_rgb = np.array(img)

        # =========================================================
        # DEEP LEARNING
        # =========================================================
        if self.model_type == "deep_learning":

            from ml.models.deep_models import preprocess_input_for

            processed = preprocess_pipeline(
                raw_rgb,
                IMAGE_SIZE,
                denoise_enabled=False,
                enhance_enabled=False,
            )

            processed_uint8 = (
                processed * 255
            ).astype("uint8")

            backbone_preprocess = preprocess_input_for(
                self.model_name
            )

            batch = backbone_preprocess(
                processed_uint8.astype("float32")
            )[np.newaxis, ...]

            proba = self.model.predict(
                batch,
                verbose=0,
            )[0]

            input_batch = batch

        # =========================================================
        # HYBRID
        # =========================================================
        elif self.model_type == "hybrid":

            from ml.models.deep_models import preprocess_input_for

            # -----------------------------------------------------
            # 1. ResNet deep features
            # -----------------------------------------------------

            processed_deep = preprocess_pipeline(
                raw_rgb,
                IMAGE_SIZE,
                denoise_enabled=False,
                enhance_enabled=False,
            )

            deep_uint8 = (
                processed_deep * 255
            ).astype("uint8")

            deep_image = deep_uint8.astype(
                "float32"
            )

            backbone_preprocess = preprocess_input_for(
                "ResNet"
            )

            deep_batch = backbone_preprocess(
                deep_image
            )[np.newaxis, ...]

            deep_features = (
                self.hybrid_feature_extractor.predict(
                    deep_batch,
                    verbose=0,
                )
            )

            # -----------------------------------------------------
            # 2. Traditional features
            # -----------------------------------------------------

            processed_traditional = preprocess_pipeline(
                raw_rgb,
                TRADITIONAL_ML_IMAGE_SIZE,
                denoise_enabled=False,
                enhance_enabled=False,
            )

            traditional_uint8 = (
                processed_traditional * 255
            ).astype("uint8")

            traditional_features = (
                extract_features_single(
                    traditional_uint8
                ).reshape(1, -1)
            )

            # -----------------------------------------------------
            # 3. Feature fusion
            # -----------------------------------------------------

            fused_features = np.concatenate(
                [
                    deep_features,
                    traditional_features,
                ],
                axis=1,
            )

            # -----------------------------------------------------
            # 4. StandardScaler
            # -----------------------------------------------------

            scaled_features = (
                self.hybrid_scaler.transform(
                    fused_features
                )
            )

            # -----------------------------------------------------
            # 5. PCA
            # -----------------------------------------------------

            pca_features = (
                self.hybrid_pca.transform(
                    scaled_features
                )
            )

            # -----------------------------------------------------
            # 6. RBF SVM
            # -----------------------------------------------------

            proba = (
                self.hybrid_svm.predict_proba(
                    pca_features
                )[0]
            )

            # Hybrid is not a direct Keras classifier,
            # so Grad-CAM cannot use the normal input_batch.
            input_batch = None

        # =========================================================
        # TRADITIONAL
        # =========================================================
        else:

            processed = preprocess_pipeline(
                raw_rgb,
                TRADITIONAL_ML_IMAGE_SIZE,
                denoise_enabled=True,
                enhance_enabled=True,
            )

            processed_uint8 = (
                processed * 255
            ).astype("uint8")

            feats = (
                extract_features_single(
                    processed_uint8
                ).reshape(1, -1)
            )

            if self.scaler is not None:
                feats = self.scaler.transform(feats)

            if self.pca is not None:
                feats = self.pca.transform(feats)

            proba = (
                self.model.predict_proba(
                    feats
                )[0]
            )

            input_batch = None

        # =========================================================
        # RESULT
        # =========================================================

        pred_idx = int(
            np.argmax(proba)
        )

        probabilities = {
            CLASS_NAMES[i]: float(proba[i])
            for i in range(len(CLASS_NAMES))
        }

        return {
            "predicted_class": CLASS_NAMES[pred_idx],
            "confidence": float(proba[pred_idx]),
            "probabilities": probabilities,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "raw_image_rgb": raw_rgb,
            "predicted_index": pred_idx,
            "input_batch": input_batch,
        }


# Module-level singleton.
engine = InferenceEngine()