import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv
import base64

# Load environment variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.title("MedVision-XR COVID-19 Predictor")
st.write("Upload an X-ray image to detect COVID-19, Lung Opacity, Viral Pneumonia, or Normal")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
show_gradcam = st.checkbox("Show Grad-CAM heatmap", value=True)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=400)

    # Convert to bytes and send to API
    bytes_data = uploaded_file.getvalue()
    files = {"file": (uploaded_file.name, bytes_data, uploaded_file.type)}

    try:
        # Add gradcam param to request
        response = requests.post(
            f"{BACKEND_URL}/predict?gradcam={str(show_gradcam).lower()}",
            files=files
        )
        response.raise_for_status()
        result = response.json()

        st.write(f"**Predicted Class:** {result['class']}")
        st.write(f"**Confidence:** {result['confidence']*100:.2f}%")

        # Display Grad-CAM image if available
        if show_gradcam and "gradcam" in result:
            gradcam_bytes = base64.b64decode(result["gradcam"])
            gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
            st.image(gradcam_img, caption="Grad-CAM Heatmap", width=400)

    except requests.exceptions.HTTPError as e:
        error_msg = ""
        if e.response is not None:
            try:
                error_msg = e.response.json().get("error", "")
            except Exception:
                error_msg = e.response.text
        st.error(f"Backend error ({e.response.status_code}): {error_msg or e}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
