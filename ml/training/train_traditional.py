"""
Trains every traditional-ML model, evaluates it on the (shared) held-out
test set, and writes metrics + the fitted model to disk.

Run: python -m ml.training.train_traditional
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from ml.config import (
    MODELS_DIR, REPORTS_DIR, TRADITIONAL_MODEL_NAMES, CLASS_NAMES, RANDOM_SEED,
)
from ml.utils.seed import set_global_seed
from ml.utils.metrics import compute_full_metrics, measure_inference_time
from ml.data.dataset_loader import discover_dataset, stratified_split, load_image_array
from ml.config import TRADITIONAL_ML_IMAGE_SIZE
from ml.features.feature_extraction import extract_features_batch, FeaturePCA
from ml.models.traditional_models import build_traditional_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_split_arrays(split_samples):
    images = [load_image_array(s.path, TRADITIONAL_ML_IMAGE_SIZE) for s in split_samples]
    labels = np.array([s.label for s in split_samples])
    return images, labels


def main():
    set_global_seed(RANDOM_SEED)
    samples = discover_dataset()
    splits = stratified_split(samples)

    logger.info("Extracting features (this touches disk once per image, then caches in memory)...")
    train_imgs, y_train = _load_split_arrays(splits.train)
    val_imgs, y_val = _load_split_arrays(splits.val)
    test_imgs, y_test = _load_split_arrays(splits.test)

    X_train_raw = extract_features_batch(train_imgs)
    X_val_raw = extract_features_batch(val_imgs)
    X_test_raw = extract_features_batch(test_imgs)

    # Scale using TRAIN statistics only (no leakage), then compress with PCA
    # fit on TRAIN only, applied identically to val/test.
    scaler = StandardScaler().fit(X_train_raw)
    X_train_s = scaler.transform(X_train_raw)
    X_val_s = scaler.transform(X_val_raw)
    X_test_s = scaler.transform(X_test_raw)

    pca = FeaturePCA(n_components=min(150, X_train_s.shape[0] - 1, X_train_s.shape[1]))
    X_train = pca.fit_transform(X_train_s)
    X_val = pca.transform(X_val_s)
    X_test = pca.transform(X_test_s)

    dump(scaler, MODELS_DIR / "traditional_feature_scaler.joblib")
    pca.save(MODELS_DIR / "traditional_feature_pca.joblib")

    # Traditional models are trained on train+val combined (they don't need a
    # held-out val set for early stopping the way deep nets do), then always
    # evaluated only on the untouched test split.
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    all_results = {}
    for name in TRADITIONAL_MODEL_NAMES:
        logger.info("=== Training %s ===", name)
        model = build_traditional_model(name)

        t0 = time.perf_counter()
        model.fit(X_trainval, y_trainval)
        training_time = time.perf_counter() - t0

        # 5-fold stratified CV on the training data, reported for academic rigor
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_trainval, y_trainval, cv=cv, scoring="f1_macro", n_jobs=-1)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = compute_full_metrics(y_test, y_pred, y_proba, CLASS_NAMES)
        inference_ms = measure_inference_time(lambda X: model.predict_proba(X), X_test)

        metrics.update({
            "model_name": name,
            "model_type": "traditional",
            "training_time_seconds": training_time,
            "inference_time_ms_per_image": inference_ms,
            "cv_f1_macro_mean": float(cv_scores.mean()),
            "cv_f1_macro_std": float(cv_scores.std()),
            "n_parameters": None,
            "model_size_kb": None,
        })

        model_path = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
        dump(model, model_path)
        metrics["model_size_kb"] = round(model_path.stat().st_size / 1024, 2)

        all_results[name] = metrics
        logger.info("%s -> macro F1=%.4f  accuracy=%.4f", name, metrics["f1_macro"], metrics["accuracy"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "traditional_models_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved traditional model results to %s", REPORTS_DIR / "traditional_models_results.json")


if __name__ == "__main__":
    main()
