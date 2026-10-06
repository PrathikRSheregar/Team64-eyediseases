"""
Central configuration for the ML pipeline.
Nothing here is hard-coded in the sense of secrets - paths/hyperparameters are
meant to be edited or overridden via environment variables for reproducibility.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(os.getenv("EYE_DATASET_DIR", ROOT_DIR / "dataset"))
MODELS_DIR = Path(os.getenv("EYE_MODELS_DIR", ROOT_DIR / "models"))
ARTIFACTS_DIR = Path(os.getenv("EYE_ARTIFACTS_DIR", ROOT_DIR / "ml" / "artifacts"))
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
FEATURES_CACHE_DIR = ARTIFACTS_DIR / "features_cache"

for d in [MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, PLOTS_DIR, FEATURES_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# The mapping name -> expected directory name. If your dataset uses different
# folder names, change this dict only (nothing else needs to change).
CLASS_DIR_MAP = {
    "Cataract": "cataract",
    "Diabetic Retinopathy": "diabetic_retinopathy",
    "Glaucoma": "glaucoma",
    "Normal": "normal",
}
CLASS_NAMES = list(CLASS_DIR_MAP.keys())          # canonical, ordered list
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}

# ---------------------------------------------------------------------------
# Image / preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # (H, W) - standard for transfer-learning backbones
TRADITIONAL_ML_IMAGE_SIZE = (128, 128)  # smaller, used before feature extraction
CHANNELS = 3
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Split / reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS_DEEP = int(os.getenv("EYE_EPOCHS", 25))
FINE_TUNE_EPOCHS = int(os.getenv("EYE_FINE_TUNE_EPOCHS", 10))
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
EARLY_STOPPING_PATIENCE = 5
LR_REDUCE_PATIENCE = 3

DEEP_MODEL_NAMES = ["MobileNet", "ResNet", "EfficientNet", "DenseNet"]
TRADITIONAL_MODEL_NAMES = [
    "Decision Tree", "Random Forest", "SVM", "KNN", "Logistic Regression"
]
ALL_MODEL_NAMES = TRADITIONAL_MODEL_NAMES + DEEP_MODEL_NAMES

# ---------------------------------------------------------------------------
# Model selection criterion (see evaluation/compare_models.py)
# ---------------------------------------------------------------------------
# Final score = weighted sum of macro metrics, NOT accuracy alone.
SELECTION_WEIGHTS = {
    "macro_f1": 0.40,
    "balanced_accuracy": 0.25,
    "macro_recall": 0.20,       # sensitivity
    "macro_roc_auc": 0.15,
}
