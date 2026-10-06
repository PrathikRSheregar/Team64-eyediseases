"""
Orchestrator: trains every traditional model + every deep model, then runs
evaluation/comparison and best-model selection.

Run: python -m ml.training.train_all
"""
import logging
from ml.training.train_traditional import main as train_traditional
from ml.training.train_deep import train_one_model
from ml.config import DEEP_MODEL_NAMES
from ml.evaluation.compare_models import build_comparison

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("STEP 1/3 - Training traditional ML models")
    train_traditional()

    logger.info("STEP 2/3 - Training deep learning models")
    for name in DEEP_MODEL_NAMES:
        train_one_model(name)

    logger.info("STEP 3/3 - Comparing all models and selecting production model")
    build_comparison()


if __name__ == "__main__":
    main()
