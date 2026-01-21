import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

import base64
from io import BytesIO
from PIL import Image

from app.utils import make_gradcam_heatmap, overlay_gradcam

# Load the model
model = load_model("models/covid_efficientnetb0_grad_cam.keras")

# Class labels (must match training)
CLASS_LABELS = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']


def predict_image(img_array, return_gradcam=False):
    """
    Predict class and optionally return Grad-CAM overlay.
    img_array: (H,W,C) or (1,H,W,C), normalized [0,1]
    """
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    img_np = np.array(img_array, dtype=np.float32)
    preds = model.predict(img_np)
    class_idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(preds[0][class_idx])
    label = CLASS_LABELS[class_idx]

    gradcam_base64 = None
    if return_gradcam:
        # Find last conv layer
        last_conv_layer_name = "top_conv"
        conv_layer = _safe_get_layer(model, last_conv_layer_name)

        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[conv_layer.output, model.output]
        )

        img_tensor = tf.convert_to_tensor(img_np)
        heatmap = make_gradcam_heatmap(img_tensor, grad_model, pred_index=class_idx)
        # Convert overlay to base64
        img_orig = _denormalize_efficientnet(img_np[0])
        gradcam_img = overlay_gradcam(img_orig, heatmap)
        pil_img = Image.fromarray(gradcam_img)
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return label, confidence, gradcam_base64


def _denormalize_efficientnet(img):
    """
    EfficientNet preprocessing maps pixels to [-1, 1] via:
    x = (x / 127.5) - 1
    Reverse that to recover uint8 image.
    """
    img = ((img + 1.0) * 127.5).clip(0, 255)
    return img.astype(np.uint8)


def _safe_get_layer(keras_model, layer_name):
    """Return layer even if nested inside sub-models."""
    try:
        return keras_model.get_layer(layer_name)
    except ValueError:
        for layer in keras_model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    return layer.get_layer(layer_name)
                except ValueError:
                    continue
        raise ValueError(f"Cannot find layer '{layer_name}' in model")
