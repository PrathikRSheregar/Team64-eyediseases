"""
Feature extraction for TRADITIONAL ML models.

Feeding raw flattened pixels (128*128*3 = 49,152 dims) directly into an SVM
or KNN is exactly the "extremely high-dimensional raw input" the project spec
warns against - it's slow, overfits easily, and ignores the fact that texture
and color statistics carry most of the discriminative signal for these
disease classes. Instead we build a compact, interpretable feature vector:

    [ HOG shape/texture features | color histogram features | resized-gray
      statistical features ] -> optionally compressed with PCA

This keeps traditional models fast to train, meaningfully different from the
CNNs (so the comparison is informative), and still gives feature_importance_
/ coefficients that can be inspected for explainability.
"""
from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from joblib import dump, load

from ml.config import TRADITIONAL_ML_IMAGE_SIZE, FEATURES_CACHE_DIR


def _hog_features(gray: np.ndarray) -> np.ndarray:
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def _color_histogram_features(rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    hists = []
    for ch in range(3):
        h = cv2.calcHist([rgb], [ch], None, [bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        hists.append(h)
    return np.concatenate(hists)


def _statistical_features(gray: np.ndarray) -> np.ndarray:
    return np.array([
        gray.mean(), gray.std(), np.median(gray),
        np.percentile(gray, 25), np.percentile(gray, 75),
    ])


def extract_features_single(img_uint8: np.ndarray) -> np.ndarray:
    """img_uint8: RGB uint8 array of arbitrary size."""
    img = cv2.resize(img_uint8, TRADITIONAL_ML_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hog_feat = _hog_features(gray)
    color_feat = _color_histogram_features(img)
    stat_feat = _statistical_features(gray)

    return np.concatenate([hog_feat, color_feat, stat_feat]).astype(np.float32)


def extract_features_batch(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([extract_features_single(img) for img in images])


class FeaturePCA:
    """Thin wrapper so the fitted PCA can be persisted alongside model artifacts
    and reused identically at inference time (fit on TRAIN split only)."""

    def __init__(self, n_components: int = 150):
        self.pca = PCA(n_components=n_components, random_state=42)
        self.fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        out = self.pca.fit_transform(X)
        self.fitted = True
        return out

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeaturePCA must be fit before calling transform()")
        return self.pca.transform(X)

    def save(self, path):
        dump(self.pca, path)
        self.fitted = True

    def load(self, path):
        self.pca = load(path)
        self.fitted = True
        return self
