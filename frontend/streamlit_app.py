import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # looks for .env file
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("MedVision-XR COVID-19 Predictor")
st.write("Upload an X-ray image to detect COVID-19, Lung Opacity, Viral Pneumonia or Normal")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    # Convert to bytes and send to API
    bytes_data = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}

    try:
        response = requests.post(f"{BACKEND_URL}/predict/", files=files)
        response.raise_for_status()
        result = response.json()
        st.write(f"**Predicted Class:** {result['class']}")
        st.write(f"**Confidence:** {result['confidence']*100:.2f}%")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
