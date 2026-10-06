"""
Trains one deep-learning / transfer-learning model end-to-end:
  1. Build tf.data pipelines from the (leak-free) train/val/test splits.
  2. Train the classification head with the backbone frozen.
  3. Fine-tune the top backbone layers at a low learning rate.
  4. Evaluate on the untouched test split.
  5. Save the best checkpoint, training history, and metrics.

Run: python -m ml.training.train_deep --model MobileNet
"""
from __future__ import annotations
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from ml.config import (
    MODELS_DIR, REPORTS_DIR, PLOTS_DIR, CLASS_NAMES, IMAGE_SIZE, BATCH_SIZE,
    EPOCHS_DEEP, FINE_TUNE_EPOCHS, LEARNING_RATE, FINE_TUNE_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, LR_REDUCE_PATIENCE, RANDOM_SEED,
)
from ml.utils.seed import set_global_seed
from ml.utils.metrics import compute_full_metrics, measure_inference_time
from ml.data.dataset_loader import discover_dataset, stratified_split
from ml.preprocessing.augmentation import build_training_augmenter
from ml.models.deep_models import build_deep_model, unfreeze_top_layers, preprocess_input_for

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_dataset(samples, preprocess_fn, augment: bool, shuffle: bool):
    paths = [str(s.path) for s in samples]
    labels = [s.label for s in samples]

    def _load(path, label):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMAGE_SIZE)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = build_training_augmenter()
        ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (preprocess_fn(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _compute_class_weights(samples):
    labels = np.array([s.label for s in samples])
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    n_classes = len(classes)
    return {int(c): float(total / (n_classes * cnt)) for c, cnt in zip(classes, counts)}


def train_one_model(model_name: str) -> dict:
    set_global_seed(RANDOM_SEED)
    samples = discover_dataset()
    splits = stratified_split(samples)
    preprocess_fn = preprocess_input_for(model_name)

    train_ds = _make_dataset(splits.train, preprocess_fn, augment=True, shuffle=True)
    val_ds = _make_dataset(splits.val, preprocess_fn, augment=False, shuffle=False)
    test_ds = _make_dataset(splits.test, preprocess_fn, augment=False, shuffle=False)
    class_weights = _compute_class_weights(splits.train)

    model = build_deep_model(model_name, freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = MODELS_DIR / f"{model_name.lower()}_best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=LR_REDUCE_PATIENCE, factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    logger.info("=== Phase 1: training head (backbone frozen) for %s ===", model_name)
    t0 = time.perf_counter()
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS_DEEP,
        class_weight=class_weights, callbacks=callbacks, verbose=2,
    )

    logger.info("=== Phase 2: fine-tuning top backbone layers for %s ===", model_name)
    model = unfreeze_top_layers(model, n_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights, callbacks=callbacks, verbose=2,
    )
    training_time = time.perf_counter() - t0

    # Merge histories for the training-curve plots
    full_history = {k: history1.history.get(k, []) + history2.history.get(k, [])
                     for k in set(history1.history) | set(history2.history)}

    model.load_weights(str(ckpt_path))

    y_true, y_pred, y_proba = [], [], []
    for batch_x, batch_y in test_ds:
        proba = model.predict(batch_x, verbose=0)
        y_proba.append(proba)
        y_pred.append(np.argmax(proba, axis=1))
        y_true.append(batch_y.numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    metrics = compute_full_metrics(y_true, y_pred, y_proba, CLASS_NAMES)
    inference_ms = measure_inference_time(
        lambda X: model.predict(X, verbose=0),
        np.stack([x for x, _ in test_ds.unbatch().take(16)]),
    )
    n_params = int(model.count_params())
    model_size_kb = round(ckpt_path.stat().st_size / 1024, 2) if ckpt_path.exists() else None

    metrics.update({
        "model_name": model_name,
        "model_type": "deep_learning",
        "training_time_seconds": training_time,
        "inference_time_ms_per_image": inference_ms,
        "n_parameters": n_params,
        "model_size_kb": model_size_kb,
        "epochs_trained": len(full_history.get("loss", [])),
    })

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / f"{model_name.lower()}_history.json", "w") as f:
        json.dump(full_history, f, indent=2)
    with open(REPORTS_DIR / f"{model_name.lower()}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_training_curves(model_name, full_history)
    logger.info("%s -> macro F1=%.4f  accuracy=%.4f", model_name, metrics["f1_macro"], metrics["accuracy"])
    return metrics


def _plot_training_curves(model_name: str, history: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.get("loss", []), label="train loss")
    axes[0].plot(history.get("val_loss", []), label="val loss")
    axes[0].set_title(f"{model_name} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train accuracy")
    axes[1].plot(history.get("val_accuracy", []), label="val accuracy")
    axes[1].set_title(f"{model_name} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{model_name.lower()}_training_curves.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["MobileNet", "ResNet", "EfficientNet", "DenseNet"])
    args = parser.parse_args()
    train_one_model(args.model)
