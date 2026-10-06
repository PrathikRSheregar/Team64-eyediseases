import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from ml.config import REPORTS_DIR, ALL_MODEL_NAMES

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    summary="List available models and which one is in production"
)
def list_models():
    meta_path = Path(settings.MODELS_DIR) / "production_model_metadata.json"

    production_model = None

    if meta_path.exists():
        production_model = json.loads(
            meta_path.read_text()
        ).get("production_model")

    return {
        "available_models": ALL_MODEL_NAMES,
        "production_model": production_model,
    }


@router.get(
    "/model-performance",
    summary="Comparison metrics for every trained model"
)
def model_performance():
    comparison_path = REPORTS_DIR / "model_comparison.json"
    traditional_path = REPORTS_DIR / "traditional_models_results.json"
    resnet_path = REPORTS_DIR / "resnet_metrics.json"
    ensemble_path = (
        REPORTS_DIR / "resnet_efficientnet_ensemble_metrics.json"
    )

    if not comparison_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="No model comparison report found yet.",
        )

    comparison = json.loads(comparison_path.read_text())

    # ---------------------------------------------------------
    # Add ResNet if its metrics file exists
    # ---------------------------------------------------------
    if resnet_path.exists():
        resnet = json.loads(resnet_path.read_text())

        # Same weighted selection formula used by the project
        selection_score = (
            0.40 * resnet["f1_macro"]
            + 0.25 * resnet["balanced_accuracy"]
            + 0.20 * resnet["recall_macro"]
            + 0.15 * resnet["roc_auc_macro"]
        )

        comparison["comparison_table"].append(
            {
                "model": "ResNet",
                "accuracy": resnet.get("accuracy"),
                "precision_macro": resnet.get("precision_macro"),
                "recall_macro": resnet.get("recall_macro"),
                "f1_macro": resnet.get("f1_macro"),
                "specificity_macro": resnet.get("specificity_macro"),
                "roc_auc_macro": resnet.get("roc_auc_macro"),
                "balanced_accuracy": resnet.get("balanced_accuracy"),
                "inference_time_ms": resnet.get("inference_time_ms"),
                "training_time_seconds": resnet.get(
                    "training_time_seconds"
                ),
                "n_parameters": resnet.get("n_parameters"),
                "model_size_kb": resnet.get("model_size_kb"),
                "selection_score": selection_score,
            }
        )

    # ---------------------------------------------------------
    # Add traditional ML models if available
    # ---------------------------------------------------------
    if traditional_path.exists():
        traditional = json.loads(traditional_path.read_text())

        for model_name, metrics in traditional.items():
            comparison["comparison_table"].append(
                {
                    "model": model_name,
                    "accuracy": metrics.get("accuracy"),
                    "precision_macro": metrics.get("precision_macro"),
                    "recall_macro": metrics.get("recall_macro"),
                    "f1_macro": metrics.get("f1_macro"),
                    "specificity_macro": metrics.get(
                        "specificity_macro"
                    ),
                    "roc_auc_macro": metrics.get(
                        "roc_auc_macro"
                    ),
                    "balanced_accuracy": metrics.get(
                        "balanced_accuracy"
                    ),
                    "inference_time_ms": metrics.get(
                        "inference_time_ms_per_image"
                    ),
                    "training_time_seconds": metrics.get(
                        "training_time_seconds"
                    ),
                    "n_parameters": metrics.get("n_parameters"),
                    "model_size_kb": metrics.get("model_size_kb"),
                    "selection_score": None,
                }
            )

    # ---------------------------------------------------------
    # Add ResNet + EfficientNet ensemble if available
    # ---------------------------------------------------------
    if ensemble_path.exists():
        ensemble_report = json.loads(
            ensemble_path.read_text()
        )

        ensemble = ensemble_report.get("ensemble")

        if ensemble:
            comparison["ensemble"] = ensemble

    # ---------------------------------------------------------
    # Recalculate best individual model
    # ---------------------------------------------------------
    valid_models = [
        row
        for row in comparison["comparison_table"]
        if row.get("selection_score") is not None
    ]

    if valid_models:
        best_model = max(
            valid_models,
            key=lambda row: row["selection_score"]
        )

        comparison["best_model"] = best_model["model"]
        comparison["best_model_score"] = best_model[
            "selection_score"
        ]

    return comparison