"""
Hybrid eye-disease classifier.

Pipeline:

Image
    -> ResNet50 deep features
    + HOG/color/statistical features
    -> feature fusion
    -> StandardScaler
    -> PCA
    -> RBF SVM
    -> 4-class prediction

Memory-efficient version:
- Images are processed in small batches.
- The entire dataset is never loaded into RAM at once.

Run:
    python -m ml.training.train_hybrid
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras.models import load_model, Model

from ml.config import (
    MODELS_DIR,
    REPORTS_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    TRADITIONAL_ML_IMAGE_SIZE,
    RANDOM_SEED,
)
from ml.utils.seed import set_global_seed
from ml.data.dataset_loader import (
    discover_dataset,
    stratified_split,
    load_image_array,
)
from ml.features.feature_extraction import extract_features_batch
from ml.models.deep_models import preprocess_input_for


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RESNET_PATH = MODELS_DIR / "resnet_best.keras"

HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_resnet_traditional_svm.joblib"
HYBRID_SCALER_PATH = MODELS_DIR / "hybrid_feature_scaler.joblib"
HYBRID_PCA_PATH = MODELS_DIR / "hybrid_feature_pca.joblib"
HYBRID_METADATA_PATH = MODELS_DIR / "hybrid_metadata.json"

PCA_COMPONENTS = 256

# Small enough for the ~3.7 GB Docker memory limit.
DEEP_BATCH_SIZE = 16
TRADITIONAL_BATCH_SIZE = 32


def build_resnet_feature_extractor():
    logger.info("Loading trained ResNet from %s", RESNET_PATH)

    resnet = load_model(
        RESNET_PATH,
        compile=False,
    )

    logger.info("ResNet layers: %d", len(resnet.layers))

    # Architecture:
    # ResNet backbone
    # -> Dropout
    # -> Dense(128)
    # -> Dropout
    # -> Dense(4)
    #
    # We use the learned Dense(128) representation.

    feature_layer = resnet.layers[-3]

    logger.info(
        "Using ResNet feature layer: %s",
        feature_layer.name,
    )

    return Model(
        inputs=resnet.input,
        outputs=feature_layer.output,
    )


def extract_resnet_features_batched(
    feature_extractor,
    samples,
):
    """
    Extract ResNet features without loading all images into RAM.
    """

    preprocess_fn = preprocess_input_for("ResNet")

    all_features = []

    total = len(samples)

    logger.info(
        "Extracting ResNet features in batches of %d...",
        DEEP_BATCH_SIZE,
    )

    for start in range(0, total, DEEP_BATCH_SIZE):
        batch_samples = samples[start:start + DEEP_BATCH_SIZE]

        images = np.stack(
            [
                load_image_array(
                    sample.path,
                    IMAGE_SIZE,
                )
                for sample in batch_samples
            ]
        ).astype(np.float32)

        images = preprocess_fn(images)

        batch_features = feature_extractor.predict(
            images,
            batch_size=DEEP_BATCH_SIZE,
            verbose=0,
        )

        all_features.append(
            np.asarray(batch_features, dtype=np.float32)
        )

        end = min(start + DEEP_BATCH_SIZE, total)

        if start == 0 or end == total or end % 160 == 0:
            logger.info(
                "ResNet features: %d/%d images",
                end,
                total,
            )

        del images
        del batch_features

    return np.concatenate(all_features, axis=0)


def extract_traditional_features_batched(samples):
    """
    Extract HOG/color/statistical features in batches.
    """

    all_features = []

    total = len(samples)

    logger.info(
        "Extracting traditional features in batches of %d...",
        TRADITIONAL_BATCH_SIZE,
    )

    for start in range(0, total, TRADITIONAL_BATCH_SIZE):
        batch_samples = samples[
            start:start + TRADITIONAL_BATCH_SIZE
        ]

        images = [
            load_image_array(
                sample.path,
                TRADITIONAL_ML_IMAGE_SIZE,
            )
            for sample in batch_samples
        ]

        batch_features = extract_features_batch(images)

        all_features.append(
            np.asarray(batch_features, dtype=np.float32)
        )

        end = min(
            start + TRADITIONAL_BATCH_SIZE,
            total,
        )

        if start == 0 or end == total or end % 320 == 0:
            logger.info(
                "Traditional features: %d/%d images",
                end,
                total,
            )

        del images
        del batch_features

    return np.concatenate(all_features, axis=0)


def extract_all_features(
    samples,
    feature_extractor,
):
    """
    Extract and fuse deep + traditional features.
    """

    logger.info("========== RESNET FEATURES ==========")

    deep_features = extract_resnet_features_batched(
        feature_extractor,
        samples,
    )

    logger.info(
        "ResNet feature shape: %s",
        deep_features.shape,
    )

    logger.info("========== TRADITIONAL FEATURES ==========")

    traditional_features = extract_traditional_features_batched(
        samples,
    )

    logger.info(
        "Traditional feature shape: %s",
        traditional_features.shape,
    )

    logger.info("========== FEATURE FUSION ==========")

    fused = np.concatenate(
        [
            deep_features,
            traditional_features,
        ],
        axis=1,
    )

    logger.info(
        "Fused feature shape: %s",
        fused.shape,
    )

    del deep_features
    del traditional_features

    return fused


def evaluate(model, X, y, name):

    y_pred = model.predict(X)

    accuracy = accuracy_score(
        y,
        y_pred,
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        y_pred,
    )

    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro",
    )

    logger.info(
        "%s accuracy = %.4f",
        name,
        accuracy,
    )

    logger.info(
        "%s balanced accuracy = %.4f",
        name,
        balanced_accuracy,
    )

    logger.info(
        "%s macro F1 = %.4f",
        name,
        macro_f1,
    )

    print("\n======================================")
    print(name)
    print("======================================")

    print(
        classification_report(
            y,
            y_pred,
            target_names=CLASS_NAMES,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y,
            y_pred,
        )
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "f1_macro": float(macro_f1),
    }


def main():

    set_global_seed(RANDOM_SEED)

    # ---------------------------------------------------------
    # DATASET
    # ---------------------------------------------------------

    logger.info("Discovering dataset...")

    samples = discover_dataset()

    splits = stratified_split(
        samples
    )

    logger.info(
        "Train: %d",
        len(splits.train),
    )

    logger.info(
        "Validation: %d",
        len(splits.val),
    )

    logger.info(
        "Test: %d",
        len(splits.test),
    )

    y_train = np.asarray(
        [s.label for s in splits.train]
    )

    y_val = np.asarray(
        [s.label for s in splits.val]
    )

    y_test = np.asarray(
        [s.label for s in splits.test]
    )

    # ---------------------------------------------------------
    # RESNET FEATURE EXTRACTOR
    # ---------------------------------------------------------

    feature_extractor = build_resnet_feature_extractor()

    # ---------------------------------------------------------
    # FEATURE EXTRACTION
    # ---------------------------------------------------------

    logger.info(
        "\n========== TRAIN FEATURES =========="
    )

    X_train = extract_all_features(
        splits.train,
        feature_extractor,
    )

    logger.info(
        "\n========== VALIDATION FEATURES =========="
    )

    X_val = extract_all_features(
        splits.val,
        feature_extractor,
    )

    logger.info(
        "\n========== TEST FEATURES =========="
    )

    X_test = extract_all_features(
        splits.test,
        feature_extractor,
    )

    # ---------------------------------------------------------
    # SCALE
    # ---------------------------------------------------------

    logger.info(
        "\n========== STANDARDIZATION =========="
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ---------------------------------------------------------
    # PCA
    # ---------------------------------------------------------

    n_components = min(
        PCA_COMPONENTS,
        X_train_scaled.shape[0] - 1,
        X_train_scaled.shape[1],
    )

    logger.info(
        "PCA components: %d",
        n_components,
    )

    pca = PCA(
        n_components=n_components,
        whiten=True,
        random_state=RANDOM_SEED,
    )

    X_train_pca = pca.fit_transform(
        X_train_scaled
    )

    X_val_pca = pca.transform(
        X_val_scaled
    )

    X_test_pca = pca.transform(
        X_test_scaled
    )

    explained_variance = float(
        np.sum(
            pca.explained_variance_ratio_
        )
    )

    logger.info(
        "PCA explained variance: %.4f",
        explained_variance,
    )

    # ---------------------------------------------------------
    # TRAIN ON TRAIN + VALIDATION
    # ---------------------------------------------------------

    X_trainval = np.concatenate(
        [
            X_train_pca,
            X_val_pca,
        ],
        axis=0,
    )

    y_trainval = np.concatenate(
        [
            y_train,
            y_val,
        ],
        axis=0,
    )

    logger.info(
        "\n========== TRAINING SVM =========="
    )

    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    t0 = time.perf_counter()

    svm.fit(
        X_trainval,
        y_trainval,
    )

    training_time = time.perf_counter() - t0

    logger.info(
        "SVM training time: %.2f seconds",
        training_time,
    )

    # ---------------------------------------------------------
    # TEST
    # ---------------------------------------------------------

    test_metrics = evaluate(
        svm,
        X_test_pca,
        y_test,
        "Hybrid ResNet + Traditional Features + SVM",
    )

    # ---------------------------------------------------------
    # SAVE ARTIFACTS
    # ---------------------------------------------------------

    logger.info(
        "\n========== SAVING =========="
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        svm,
        HYBRID_MODEL_PATH,
    )

    joblib.dump(
        scaler,
        HYBRID_SCALER_PATH,
    )

    joblib.dump(
        pca,
        HYBRID_PCA_PATH,
    )

    metadata = {
        "model_name": "Hybrid ResNet + Traditional Features + RBF SVM",
        "deep_feature_extractor": "resnet_best.keras",
        "deep_feature_dimensions": 128,
        "fused_feature_dimensions": int(
            X_train.shape[1]
        ),
        "pca_components": int(
            n_components
        ),
        "pca_explained_variance": explained_variance,
        "svm_kernel": "rbf",
        "svm_C": 10,
        "svm_gamma": "scale",
        "class_weight": "balanced",
        "test_metrics": test_metrics,
        "training_time_seconds": training_time,
        "classes": CLASS_NAMES,
    }

    with open(
        HYBRID_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    logger.info(
        "Saved: %s",
        HYBRID_MODEL_PATH,
    )

    logger.info(
        "Saved: %s",
        HYBRID_SCALER_PATH,
    )

    logger.info(
        "Saved: %s",
        HYBRID_PCA_PATH,
    )

    logger.info(
        "Saved: %s",
        HYBRID_METADATA_PATH,
    )


if __name__ == "__main__":
    main()