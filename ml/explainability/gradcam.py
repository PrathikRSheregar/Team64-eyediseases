"""
Grad-CAM for the deep-learning models (Keras).

Given an input image and the trained model, produces a heatmap over the
image showing which spatial regions most influenced the predicted class -
useful for sanity-checking the model and for the frontend's explainability
view. This is NOT a medical finding; it only reflects what pixels the model
weighted most heavily for its own decision.
"""
from __future__ import annotations
import numpy as np
import tensorflow as tf
import cv2


def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model,
                          backbone_layer_index: int, last_conv_layer_name: str,
                          pred_index: int | None = None) -> np.ndarray:
    """
    img_array: preprocessed batch of shape (1, H, W, 3) - already run through
               the same backbone-specific preprocess_input used at training time.
    model: the full functional model (Input -> backbone -> head).
    backbone_layer_index: index of the backbone sub-model within `model.layers`
                           (1 in our architecture: Input=0, backbone=1, ...).
    """
    backbone = model.layers[backbone_layer_index]
    conv_layer = backbone.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=backbone.input, outputs=[conv_layer.output, backbone.output]
    )

    # Rebuild the head as a separate callable on the pooled backbone output
    head_input = tf.keras.Input(shape=backbone.output_shape[1:])
    x = head_input
    for layer in model.layers[backbone_layer_index + 1:]:
        x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    with tf.GradientTape() as tape:
        conv_output, pooled_output = grad_model(img_array)
        tape.watch(conv_output)
        predictions = head_model(pooled_output)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_rgb_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resizes the heatmap to the original image size and overlays it as a
    JET colormap, returning an RGB uint8 array ready to encode/send."""
    h, w = original_rgb_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def traditional_feature_importance(model, feature_names: list[str] | None = None, top_k: int = 20) -> dict:
    """For traditional ML models exposing feature_importances_ (tree-based)
    or coef_ (linear models), returns the top-k most influential features."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        return {"available": False, "reason": "Model does not expose feature importances (e.g. SVM/KNN)."}

    idx = np.argsort(importances)[::-1][:top_k]
    names = feature_names or [f"feature_{i}" for i in range(len(importances))]
    return {
        "available": True,
        "top_features": [{"feature": names[i], "importance": float(importances[i])} for i in idx],
    }
