"""
Local filesystem image storage, structured so it maps 1:1 onto an object-store
key scheme (user_id/prediction_id/filename) - swapping this module's body for
an S3/Cloudinary/MinIO client later requires no changes to callers.
"""
from __future__ import annotations
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import UPLOAD_PATH


def _safe_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return ext if ext in allowed else ".jpg"


def build_storage_key(user_id: str, prediction_id: str, original_filename: str) -> str:
    """user_id/prediction_id/uuid.ext - collision-proof, no path traversal risk
    since we never use the original filename directly on disk."""
    ext = _safe_extension(original_filename)
    return f"{user_id}/{prediction_id}/{uuid.uuid4().hex}{ext}"


def save_bytes(key: str, data: bytes) -> str:
    path = UPLOAD_PATH / key
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return str(path)


def resolve_path(key: str) -> Path:
    return UPLOAD_PATH / key


def build_gradcam_key(user_id: str, prediction_id: str) -> str:
    return f"{user_id}/{prediction_id}/gradcam.jpg"
