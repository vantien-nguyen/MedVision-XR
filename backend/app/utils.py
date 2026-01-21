import io
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

IMG_SIZE = (224, 224)


def preprocess_image(image_bytes: bytes):
    """Load raw bytes, resize, and apply model preprocessing."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    # Ensure downstream consumers always receive numpy arrays
    if isinstance(img_array, tf.Tensor):
        img_array = img_array.numpy()
    return img_array


# Grad-CAM helper
def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    if not isinstance(img_array, tf.Tensor):
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    else:
        img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_array)
        outputs = grad_model(img_array)
        if isinstance(outputs, (list, tuple)) and len(outputs) == 2:
            conv_outputs, predictions = outputs
        else:
            raise ValueError("Grad-CAM model must return [conv_outputs, predictions]")

        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return np.clip(overlayed, 0, 255).astype(np.uint8)
