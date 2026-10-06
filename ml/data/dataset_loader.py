"""
Dataset discovery, validation, and stratified train/val/test splitting.

Expects (by default):

    dataset/
      cataract/*.jpg
      diabetic_retinopathy/*.jpg
      glaucoma/*.jpg
      normal/*.jpg

Folder names are configurable via ml/config.py::CLASS_DIR_MAP - nothing else
in the pipeline needs to change if your folder names differ.
"""
from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError

from ml.config import (
    DATASET_DIR, CLASS_DIR_MAP, CLASS_TO_IDX, VALID_EXTENSIONS,
    RANDOM_SEED, TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    path: Path
    class_name: str
    label: int


@dataclass
class DatasetSplits:
    train: List[DatasetSample] = field(default_factory=list)
    val: List[DatasetSample] = field(default_factory=list)
    test: List[DatasetSample] = field(default_factory=list)


def _is_valid_image(path: Path) -> bool:
    """Open + verify the image is a real, decodable, non-corrupt image file."""
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _file_hash(path: Path, block_size: int = 65536) -> str:
    """MD5 of file bytes, used for practical (exact-duplicate) detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_dataset(dataset_dir: Path = DATASET_DIR, validate: bool = True,
                      drop_duplicates: bool = True) -> List[DatasetSample]:
    """
    Walk the dataset directory, validate images, drop exact duplicates
    (by content hash), and return a flat list of samples.
    """
    samples: List[DatasetSample] = []
    seen_hashes: Dict[str, Path] = {}
    n_invalid = 0
    n_duplicate = 0

    for class_name, dir_name in CLASS_DIR_MAP.items():
        class_dir = dataset_dir / dir_name
        if not class_dir.exists():
            logger.warning("Class directory not found: %s (expected for '%s')", class_dir, class_name)
            continue

        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if validate and not _is_valid_image(path):
                n_invalid += 1
                continue
            if drop_duplicates:
                try:
                    h = _file_hash(path)
                except OSError:
                    n_invalid += 1
                    continue
                if h in seen_hashes:
                    n_duplicate += 1
                    continue
                seen_hashes[h] = path
            samples.append(DatasetSample(path=path, class_name=class_name, label=CLASS_TO_IDX[class_name]))

    logger.info(
        "Discovered %d valid images (%d invalid skipped, %d duplicates skipped)",
        len(samples), n_invalid, n_duplicate,
    )
    if not samples:
        raise RuntimeError(
            f"No images found under {dataset_dir}. Expected subfolders: "
            f"{list(CLASS_DIR_MAP.values())}. Place your dataset there or set "
            f"the EYE_DATASET_DIR environment variable."
        )
    return samples


def stratified_split(
    samples: List[DatasetSample],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> DatasetSplits:
    """
    Stratified split so each class keeps the same proportion in train/val/test.
    No image ever appears in more than one split (split is on the sample list,
    not on augmented copies, so there is no leakage).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "ratios must sum to 1"

    labels = [s.label for s in samples]
    train, temp = train_test_split(
        samples, train_size=train_ratio, random_state=seed, stratify=labels
    )
    remaining_ratio = val_ratio + test_ratio
    val_size = val_ratio / remaining_ratio
    temp_labels = [s.label for s in temp]
    val, test = train_test_split(
        temp, train_size=val_size, random_state=seed, stratify=temp_labels
    )
    return DatasetSplits(train=train, val=val, test=test)


def class_distribution(samples: List[DatasetSample]) -> Dict[str, int]:
    dist = {name: 0 for name in CLASS_DIR_MAP.keys()}
    for s in samples:
        dist[s.class_name] += 1
    return dist


def dataset_statistics(splits: DatasetSplits) -> dict:
    """Structured stats consumed by both the CLI report and the backend
    GET /api/dataset/statistics endpoint."""
    all_samples = splits.train + splits.val + splits.test
    total = len(all_samples)
    per_class_total = class_distribution(all_samples)
    return {
        "total_images": total,
        "images_per_class": per_class_total,
        "class_percentages": {
            k: round(100 * v / total, 2) if total else 0 for k, v in per_class_total.items()
        },
        "train_images": len(splits.train),
        "val_images": len(splits.val),
        "test_images": len(splits.test),
        "train_per_class": class_distribution(splits.train),
        "val_per_class": class_distribution(splits.val),
        "test_per_class": class_distribution(splits.test),
    }


def load_image_array(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Load an image as an RGB uint8 array resized to `size` (H, W)."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size[1], size[0]))
        return np.array(img, dtype=np.uint8)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    samples = discover_dataset()
    splits = stratified_split(samples)
    stats = dataset_statistics(splits)
    import json
    print(json.dumps(stats, indent=2))
