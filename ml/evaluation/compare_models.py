"""
Aggregates every model's result JSON (written by train_traditional.py and
train_deep.py), builds the comparison table + charts, and selects the
production model using a defined, non-accuracy-only criterion.

Run: python -m ml.evaluation.compare_models
"""
from __future__ import annotations
import json
import logging
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ml.config import (
    REPORTS_DIR, PLOTS_DIR, MODELS_DIR, CLASS_NAMES,
    ALL_MODEL_NAMES, DEEP_MODEL_NAMES, SELECTION_WEIGHTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_all_results() -> dict:
    results = {}

    trad_path = REPORTS_DIR / "traditional_models_results.json"
    if trad_path.exists():
        results.update(json.loads(trad_path.read_text()))

    for name in DEEP_MODEL_NAMES:
        p = REPORTS_DIR / f"{name.lower()}_metrics.json"
        if p.exists():
            results[name] = json.loads(p.read_text())

    if not results:
        raise RuntimeError(
            "No model result files found. Run ml/training/train_all.py "
            "(or the individual train_traditional.py / train_deep.py scripts) first."
        )
    return results


def _selection_score(metrics: dict) -> float:
    """Weighted composite score - defined in ml/config.py::SELECTION_WEIGHTS.
    Deliberately NOT accuracy alone: it combines macro F1 (balances all four
    classes equally regardless of class size), balanced accuracy, macro
    recall/sensitivity (clinically important - missing a real disease case is
    worse than a false alarm), and macro ROC-AUC (ranking quality)."""
    roc = metrics.get("roc_auc_macro") or 0.0
    return (
        SELECTION_WEIGHTS["macro_f1"] * metrics["f1_macro"]
        + SELECTION_WEIGHTS["balanced_accuracy"] * metrics["balanced_accuracy"]
        + SELECTION_WEIGHTS["macro_recall"] * metrics["recall_macro"]
        + SELECTION_WEIGHTS["macro_roc_auc"] * roc
    )


def _bar_chart(results: dict, metric_key: str, title: str, filename: str):
    names = list(results.keys())
    values = [results[n].get(metric_key) or 0 for n in names]
    plt.figure(figsize=(10, 5))
    colors = ["#4C6EF5" if n not in DEEP_MODEL_NAMES else "#12B886" for n in names]
    bars = plt.bar(names, values, color=colors)
    plt.title(title)
    plt.ylabel(metric_key)
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.close()


def _confusion_matrix_plots(results: dict):
    for name, metrics in results.items():
        cm = np.array(metrics.get("confusion_matrix", []))
        if cm.size == 0:
            continue
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        safe_name = name.replace(" ", "_").lower()
        plt.savefig(PLOTS_DIR / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close()


def build_comparison():
    results = _load_all_results()

    for key, title, fname in [
        ("accuracy", "Accuracy Comparison", "compare_accuracy.png"),
        ("precision_macro", "Precision (macro) Comparison", "compare_precision.png"),
        ("recall_macro", "Recall / Sensitivity (macro) Comparison", "compare_recall.png"),
        ("f1_macro", "F1-score (macro) Comparison", "compare_f1.png"),
        ("roc_auc_macro", "ROC-AUC (macro) Comparison", "compare_roc_auc.png"),
    ]:
        _bar_chart(results, key, title, fname)

    _confusion_matrix_plots(results)

    # Final comparison table
    table_rows = []
    for name in ALL_MODEL_NAMES:
        if name not in results:
            continue
        m = results[name]
        table_rows.append({
            "model": name,
            "accuracy": round(m["accuracy"], 4),
            "precision_macro": round(m["precision_macro"], 4),
            "recall_macro": round(m["recall_macro"], 4),
            "f1_macro": round(m["f1_macro"], 4),
            "specificity_macro": round(m["specificity_macro"], 4),
            "roc_auc_macro": round(m["roc_auc_macro"], 4) if m.get("roc_auc_macro") is not None else None,
            "balanced_accuracy": round(m["balanced_accuracy"], 4),
            "inference_time_ms": round(m["inference_time_ms_per_image"], 3),
            "training_time_seconds": round(m["training_time_seconds"], 2),
            "n_parameters": m.get("n_parameters"),
            "model_size_kb": m.get("model_size_kb"),
            "selection_score": round(_selection_score(m), 4),
        })

    table_rows.sort(key=lambda r: r["selection_score"], reverse=True)
    best = table_rows[0]

    comparison_report = {
        "selection_criterion": (
            "Weighted composite score = "
            f"{SELECTION_WEIGHTS['macro_f1']}*macro_F1 + "
            f"{SELECTION_WEIGHTS['balanced_accuracy']}*balanced_accuracy + "
            f"{SELECTION_WEIGHTS['macro_recall']}*macro_recall(sensitivity) + "
            f"{SELECTION_WEIGHTS['macro_roc_auc']}*macro_ROC_AUC. "
            "Accuracy alone is NOT used because the four classes are not "
            "guaranteed to be balanced; a model that only predicts the "
            "majority class would score highly on accuracy while failing "
            "the minority disease classes it matters most to detect."
        ),
        "comparison_table": table_rows,
        "best_model": best["model"],
        "best_model_score": best["selection_score"],
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison_report, f, indent=2)

    _promote_best_model(best["model"])
    logger.info("Best model selected: %s (score=%.4f)", best["model"], best["selection_score"])
    return comparison_report


def _promote_best_model(model_name: str):
    """Copies/marks the winning model as the production model the backend
    loads at startup, and writes metadata describing it."""
    production_meta = {"production_model": model_name}

    if model_name in DEEP_MODEL_NAMES:
        src = MODELS_DIR / f"{model_name.lower()}_best.keras"
        dst = MODELS_DIR / "production_model.keras"
        if src.exists():
            shutil.copy(src, dst)
        production_meta["model_type"] = "deep_learning"
        production_meta["artifact"] = "production_model.keras"
    else:
        src = MODELS_DIR / f"{model_name.replace(' ', '_').lower()}.joblib"
        dst = MODELS_DIR / "production_model.joblib"
        if src.exists():
            shutil.copy(src, dst)
        production_meta["model_type"] = "traditional"
        production_meta["artifact"] = "production_model.joblib"

    with open(MODELS_DIR / "production_model_metadata.json", "w") as f:
        json.dump(production_meta, f, indent=2)


if __name__ == "__main__":
    build_comparison()
