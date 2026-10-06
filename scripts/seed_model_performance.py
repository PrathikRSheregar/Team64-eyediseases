#!/usr/bin/env python
"""
Reads ml/artifacts/reports/model_comparison.json (produced by
ml/evaluation/compare_models.py) and upserts rows into the model_performance
table, marking the selected production model with is_production=True.

Run from repo root: python scripts/seed_model_performance.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.db.session import SessionLocal
from app.models.model_performance import ModelPerformance
from ml.config import REPORTS_DIR


def main():
    report_path = REPORTS_DIR / "model_comparison.json"
    if not report_path.exists():
        raise SystemExit(f"{report_path} not found - run ml/training/train_all.py first.")

    report = json.loads(report_path.read_text())
    db = SessionLocal()
    try:
        db.query(ModelPerformance).delete()
        for row in report["comparison_table"]:
            db.add(ModelPerformance(
                model_name=row["model"],
                accuracy=row["accuracy"],
                precision=row["precision_macro"],
                recall=row["recall_macro"],
                f1_score=row["f1_macro"],
                specificity=row["specificity_macro"],
                roc_auc=row["roc_auc_macro"],
                inference_time_ms=row["inference_time_ms"],
                training_time_seconds=row["training_time_seconds"],
                n_parameters=row["n_parameters"],
                model_size_kb=row["model_size_kb"],
                is_production=(row["model"] == report["best_model"]),
            ))
        db.commit()
        print(f"Seeded {len(report['comparison_table'])} model performance rows. "
              f"Production model: {report['best_model']}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
