import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------
@st.cache_resource
def load_brain_tumor_model():
    return load_model("brain_tumor_model.h5")


model = load_brain_tumor_model()

# Class names
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# --------------------------------------------------
# Application UI
# --------------------------------------------------
st.title("🧠 Brain Tumor Detection using AI")

st.write(
    "Upload a brain MRI image and the AI model will predict "
    "the tumor type."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if uploaded_file is not None:

    try:
        # Open uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display uploaded image
        st.image(
            image,
            caption="Uploaded MRI Image",
            width="stretch"
        )

        # Resize image to model input size
        img = image.resize((224, 224))

        # Convert image to NumPy array
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array, verbose=0)

        # Get predicted class
        predicted_class = int(np.argmax(predictions[0]))

        # Get confidence
        confidence = float(np.max(predictions[0]) * 100)

        # --------------------------------------------------
        # Display Prediction
        # --------------------------------------------------
        st.subheader("Prediction")

        st.success(
            f"Predicted Tumor Type: "
            f"{class_names[predicted_class].capitalize()}"
        )

        st.write(
            f"Confidence: {confidence:.2f}%"
        )

        # --------------------------------------------------
        # Class Probabilities
        # --------------------------------------------------
        st.subheader("Class Probabilities")

        for i, prob in enumerate(predictions[0]):
            st.write(
                f"{class_names[i].capitalize()}: "
                f"{prob * 100:.2f}%"
            )

    except Exception as e:
        st.error(
            "An error occurred while processing the MRI image."
        )
        st.exception(e)
