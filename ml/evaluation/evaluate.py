"""
Standalone re-evaluation of a single already-trained model on the test split
(useful for regenerating a report/plots without retraining).

Run: python -m ml.evaluation.evaluate --model MobileNet
"""
import argparse
import json
import logging

from ml.config import REPORTS_DIR, DEEP_MODEL_NAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    if args.model in DEEP_MODEL_NAMES:
        path = REPORTS_DIR / f"{args.model.lower()}_metrics.json"
    else:
        path = REPORTS_DIR / "traditional_models_results.json"

    if not path.exists():
        raise SystemExit(f"No results found for {args.model}. Train it first.")

    data = json.loads(path.read_text())
    metrics = data[args.model] if args.model in data else data
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
