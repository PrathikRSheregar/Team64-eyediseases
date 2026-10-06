"""Re-exports the SAME preprocessing pipeline used during training (ml/ package)
so a live prediction request is treated identically to a training-time image.
The backend and ml packages are siblings in the repo so this import works
when the backend is run with the repo root on PYTHONPATH (see Dockerfile/README)."""
from ml.preprocessing.image_preprocessing import preprocess_pipeline  # noqa: F401
from ml.features.feature_extraction import extract_features_single  # noqa: F401
from ml.config import IMAGE_SIZE, TRADITIONAL_ML_IMAGE_SIZE, CLASS_NAMES  # noqa: F401
