"""
Data augmentation - applied ONLY to the training split, never to
validation/test, to avoid inflating reported performance.

Transformations are intentionally conservative: eye/fundus images have
diagnostically meaningful left/right and up/down orientation cues (e.g.
optic-disc position, vessel pattern), so we avoid vertical flips and large
rotations that could turn a plausible eye image into an implausible one.
"""
import tensorflow as tf


def build_training_augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),          # left/right eye symmetry - safe
        tf.keras.layers.RandomRotation(0.05),               # ~18 degrees max - small, realistic
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
    ], name="training_augmentation")
