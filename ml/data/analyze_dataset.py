"""
Generates dataset statistics + a class-distribution / split-distribution plot.
Run: python -m ml.data.analyze_dataset
"""
import json
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml.config import PLOTS_DIR, REPORTS_DIR
from ml.data.dataset_loader import discover_dataset, stratified_split, dataset_statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_class_distribution(stats: dict, out_path):
    classes = list(stats["images_per_class"].keys())
    counts = list(stats["images_per_class"].values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(classes, counts, color=["#4C6EF5", "#F76707", "#12B886", "#748FFC"])
    plt.title("Class Distribution")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=15)
    for b, c in zip(bars, counts):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), str(c), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_split_distribution(stats: dict, out_path):
    classes = list(stats["train_per_class"].keys())
    train = [stats["train_per_class"][c] for c in classes]
    val = [stats["val_per_class"][c] for c in classes]
    test = [stats["test_per_class"][c] for c in classes]

    x = range(len(classes))
    width = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], train, width=width, label="Train")
    plt.bar(list(x), val, width=width, label="Validation")
    plt.bar([i + width for i in x], test, width=width, label="Test")
    plt.xticks(list(x), classes, rotation=15)
    plt.ylabel("Number of Images")
    plt.title("Train / Validation / Test Split per Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    samples = discover_dataset()
    splits = stratified_split(samples)
    stats = dataset_statistics(splits)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "dataset_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    plot_class_distribution(stats, PLOTS_DIR / "class_distribution.png")
    plot_split_distribution(stats, PLOTS_DIR / "split_distribution.png")

    logger.info("Dataset statistics written to %s", REPORTS_DIR / "dataset_statistics.json")
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
