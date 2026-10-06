"""
Shared metric computation used by every model's evaluation step, so that
traditional-ML and deep-learning models are scored identically and fairly.
"""
from __future__ import annotations
import time
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    classification_report,
)


def compute_full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list,
) -> Dict[str, Any]:
    """
    y_true: (N,) int labels
    y_pred: (N,) int labels
    y_proba: (N, C) predicted probabilities, needed for ROC-AUC
    """
    n_classes = len(class_names)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)  # = macro sensitivity
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Per-class specificity: TN / (TN + FP) computed from the confusion matrix
    specificity_per_class = []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
    macro_specificity = float(np.mean(specificity_per_class))

    try:
        macro_roc_auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=list(range(n_classes))
        )
    except ValueError:
        macro_roc_auc = float("nan")

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "specificity_macro": macro_specificity,
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "roc_auc_macro": float(macro_roc_auc) if macro_roc_auc == macro_roc_auc else None,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "per_class_specificity": {class_names[i]: specificity_per_class[i] for i in range(n_classes)},
    }


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start


def measure_inference_time(predict_fn, X, n_repeats: int = 20) -> float:
    """Average per-sample inference time in ms, for fair cross-model comparison."""
    sample = X[: min(len(X), 16)]
    predict_fn(sample)  # warmup
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        predict_fn(sample)
        times.append((time.perf_counter() - t0) / len(sample))
    return float(np.mean(times) * 1000.0)
