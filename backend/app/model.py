import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("models/covid_efficientnetb0.keras")

# Class labels (must match training)
CLASS_LABELS = ['COVID', 'Lung Opacity', 'Normal', 'Viral Pneumonia']

def predict_image(img_array):
    """
    Predict the class of the image.
    img_array should have shape (1, height, width, channels)
    """
    # Ensure input has batch dimension
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    pred = model.predict(img_array)  # shape: (1, num_classes)
    class_idx = int(np.argmax(pred, axis=1)[0])  # convert to int
    confidence = float(pred[0][class_idx])

    return CLASS_LABELS[class_idx], confidence
