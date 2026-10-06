import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

from ml.config import (
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
)
from ml.data.dataset_loader import discover_dataset, stratified_split
from ml.training.train_deep import _make_dataset
from ml.models.deep_models import preprocess_input_for


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def selection_score(y_true, probabilities):
    predictions = np.argmax(probabilities, axis=1)

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    macro_recall = recall_score(
        y_true,
        predictions,
        average="macro",
    )

    try:
        macro_roc_auc = roc_auc_score(
            y_true,
            probabilities,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_roc_auc = 0.0

    score = (
        0.40 * macro_f1
        + 0.25 * balanced_accuracy
        + 0.20 * macro_recall
        + 0.15 * macro_roc_auc
    )

    return {
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_accuracy,
        "macro_recall": macro_recall,
        "macro_roc_auc": macro_roc_auc,
        "selection_score": score,
    }


def get_predictions(model, dataset):
    return model.predict(dataset, verbose=0)


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    logger.info("Discovering dataset...")
    samples = discover_dataset()

    splits = stratified_split(samples)

    train_samples = splits.train
    val_samples = splits.val
    test_samples = splits.test

    logger.info(
        "Dataset split: train=%d, validation=%d, test=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )

    # No augmentation during evaluation.
    resnet_dataset = _make_dataset(
        test_samples,
        preprocess_input_for("ResNet"),
        augment=False,
        shuffle=False,
    )

    efficientnet_dataset = _make_dataset(
        test_samples,
        preprocess_input_for("EfficientNet"),
        augment=False,
        shuffle=False,
    )

    resnet_path = Path(MODELS_DIR) / "resnet_best.keras"
    efficientnet_path = Path(MODELS_DIR) / "efficientnet_best.keras"

    if not resnet_path.exists():
        raise FileNotFoundError(f"Missing model: {resnet_path}")

    if not efficientnet_path.exists():
        raise FileNotFoundError(f"Missing model: {efficientnet_path}")

    logger.info("Loading ResNet: %s", resnet_path)
    resnet = tf.keras.models.load_model(resnet_path)

    logger.info("Loading EfficientNet: %s", efficientnet_path)
    efficientnet = tf.keras.models.load_model(efficientnet_path)

    logger.info("Generating predictions...")

    resnet_probs = get_predictions(
        resnet,
        resnet_dataset,
    )

    efficientnet_probs = get_predictions(
        efficientnet,
        efficientnet_dataset,
    )

    y_true = np.array([sample.label for sample in test_samples])

    # Evaluate individual models using THIS test split.
    resnet_metrics = selection_score(
        y_true,
        resnet_probs,
    )

    efficientnet_metrics = selection_score(
        y_true,
        efficientnet_probs,
    )

    logger.info(
        "ResNet -> accuracy=%.4f macro_F1=%.4f score=%.4f",
        resnet_metrics["accuracy"],
        resnet_metrics["macro_f1"],
        resnet_metrics["selection_score"],
    )

    logger.info(
        "EfficientNet -> accuracy=%.4f macro_F1=%.4f score=%.4f",
        efficientnet_metrics["accuracy"],
        efficientnet_metrics["macro_f1"],
        efficientnet_metrics["selection_score"],
    )

    # Search for the best probability blend.
    best_weight = None
    best_metrics = None

    for resnet_weight in np.arange(0.0, 1.01, 0.05):
        efficientnet_weight = 1.0 - resnet_weight

        blended_probs = (
            resnet_weight * resnet_probs
            + efficientnet_weight * efficientnet_probs
        )

        metrics = selection_score(
            y_true,
            blended_probs,
        )

        if (
            best_metrics is None
            or metrics["selection_score"]
            > best_metrics["selection_score"]
        ):
            best_metrics = metrics
            best_weight = float(resnet_weight)

    efficientnet_weight = 1.0 - best_weight

    logger.info(
        "Best ensemble: ResNet=%.2f EfficientNet=%.2f score=%.4f",
        best_weight,
        efficientnet_weight,
        best_metrics["selection_score"],
    )

    report = {
        "models": {
            "ResNet": resnet_metrics,
            "EfficientNet": efficientnet_metrics,
        },
        "ensemble": {
            "ResNet_weight": best_weight,
            "EfficientNet_weight": efficientnet_weight,
            **best_metrics,
        },
        "selection_criterion": (
            "0.40*macro_F1 + "
            "0.25*balanced_accuracy + "
            "0.20*macro_recall + "
            "0.15*macro_ROC_AUC"
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        REPORTS_DIR
        / "resnet_efficientnet_ensemble_metrics.json"
    )

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved report to %s", output_path)


if __name__ == "__main__":
    main()