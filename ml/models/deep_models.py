"""
Deep learning / transfer-learning model factory (Keras/TensorFlow).

Every backbone is pretrained on ImageNet, frozen initially, topped with a
custom classification head, then fine-tuned (see ml/training/train_deep.py).

During fine-tuning, BatchNormalization layers remain frozen to preserve
their pretrained statistics and improve stability on the medical-image
dataset.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0, DenseNet121,
)
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

from ml.config import IMAGE_SIZE, NUM_CLASSES


_BACKBONES = {
    "MobileNet": (MobileNetV2, mobilenet_preprocess, "out_relu"),
    "ResNet": (ResNet50, resnet_preprocess, "conv5_block3_out"),
    "EfficientNet": (EfficientNetB0, efficientnet_preprocess, "top_activation"),
    "DenseNet": (DenseNet121, densenet_preprocess, "relu"),
}


def preprocess_input_for(name: str):
    """Returns the backbone-specific ImageNet preprocessing function."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown deep model: {name}")

    return _BACKBONES[name][1]


def last_conv_layer_name(name: str) -> str:
    """Name of the final conv/activation layer, used by Grad-CAM."""
    if name not in _BACKBONES:
        raise ValueError(f"Unknown deep model: {name}")

    return _BACKBONES[name][2]


def build_deep_model(
    name: str,
    num_classes: int = NUM_CLASSES,
    input_shape=(*IMAGE_SIZE, 3),
    freeze_backbone: bool = True,
) -> Model:

    if name not in _BACKBONES:
        raise ValueError(f"Unknown deep model: {name}")

    backbone_cls, _, _ = _BACKBONES[name]

    backbone = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        pooling="avg",
    )

    # Phase 1:
    # Entire pretrained backbone is frozen.
    backbone.trainable = not freeze_backbone

    inputs = layers.Input(shape=input_shape)

    # Always run the pretrained backbone in inference mode.
    # This prevents BatchNormalization moving statistics from changing.
    x = backbone(inputs, training=False)

    # Custom classification head
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs,
        outputs,
        name=f"{name}_eye_disease",
    )

    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30):
    """
    Fine-tuning phase.

    Unfreezes only the last n_layers of the pretrained backbone while
    keeping all BatchNormalization layers frozen.

    Earlier backbone layers remain frozen so that general ImageNet
    features such as edges and textures are preserved.
    """

    # The backbone is the first sub-model after the Input layer.
    backbone = model.layers[1]

    # Make the backbone available for selective fine-tuning.
    backbone.trainable = True

    # First freeze EVERYTHING in the backbone.
    for layer in backbone.layers:
        layer.trainable = False

    # Then selectively unfreeze the top layers.
    for layer in backbone.layers[-n_layers:]:

        # Keep BatchNormalization frozen.
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    return model