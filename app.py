import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")

# Load model
model = load_model("brain_tumor_model.h5")


class_names = ["glioma", "meningioma", "notumor", "pituitary"]
st.title("🧠 Brain Tumor Detection using AI")

st.write("Upload a brain MRI image and the AI model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)*100

    st.subheader("Prediction")

    st.success(f"Predicted Tumor Type: {class_names[predicted_class]}")

    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Class Probabilities")

    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")