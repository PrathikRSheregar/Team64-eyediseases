"""
Core image preprocessing shared by training and the live inference API,
so a prediction request is preprocessed EXACTLY like training data
(this consistency is what makes the served model trustworthy).
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


def denoise(img: np.ndarray) -> np.ndarray:
    """Mild edge-preserving denoise - retinal images are often JPEG-compressed
    with visible artifacts; heavy denoising would blur diagnostic detail, so
    this uses a small-radius bilateral filter."""
    return cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel (LAB space) - a standard, clinically-reasonable
    contrast enhancement for fundus/eye images that avoids over-amplifying
    color noise the way global histogram equalization on RGB would."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def resize(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size = (H, W)"""
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)


def normalize(img: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] float32. Backbone-specific mean/std normalization
    (e.g. ImageNet stats) is applied inside each deep model's preprocess_input,
    kept separate here so this function stays backbone-agnostic."""
    return img.astype(np.float32) / 255.0


def preprocess_pipeline(
    img: np.ndarray,
    target_size: tuple[int, int],
    denoise_enabled: bool = True,
    enhance_enabled: bool = True,
) -> np.ndarray:
    """
    Full deterministic preprocessing pipeline applied to every image
    (train, validation, test, and every live prediction request):
        resize -> denoise -> contrast enhancement -> normalize
    Returns float32 array in [0, 1], shape (H, W, 3).
    """
    out = resize(img, target_size)
    if denoise_enabled:
        out = denoise(out)
    if enhance_enabled:
        out = enhance_contrast(out)
    out = normalize(out)
    return out


def load_and_preprocess(path_or_bytes, target_size: tuple[int, int]) -> np.ndarray:
    """Accepts a filesystem path or raw bytes (from an uploaded file)."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(path_or_bytes)).convert("RGB")
    else:
        img = Image.open(path_or_bytes).convert("RGB")
    arr = np.array(img)
    return preprocess_pipeline(arr, target_size)
