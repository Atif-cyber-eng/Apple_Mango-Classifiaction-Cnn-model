import streamlit as st
import tensorflow as tf
import gdown, os, zipfile
from PIL import Image
import numpy as np

st.set_page_config(page_title="Apple vs Mango Classifier 🍎🥭")

@st.cache_resource
def load_model():
    FILE_ID = "1EmbUNvE7Y7IwqPFwrqEyFqdlvpJ7bo-i"
    ZIP_FILE = "fruits_model.zip"
    MODEL_DIR = "fruits_model"

    if not os.path.exists(MODEL_DIR):
        st.info("📦 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)
        with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
            zip_ref.extractall(".")
        st.success("✅ Model extracted!")

    model = tf.keras.models.load_model(MODEL_DIR)
    return model


st.title("🍎🥭 Fruit Classifier (Apple vs Mango)")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a fruit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    preds = model.predict(img_array)
    label = "Mango 🥭" if preds[0][0] > 0.5 else "Apple 🍎"

    st.success(f"Prediction: **{label}** (Confidence: {preds[0][0]:.2f})")
else:
    st.info("Please upload an image to classify.")
