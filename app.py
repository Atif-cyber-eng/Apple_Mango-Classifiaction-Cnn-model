import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import gdown
import os

st.set_page_config(page_title="Apple vs Mango Classifier", layout="centered")
st.title("🍎 Apple vs 🥭 Mango — Image Classifier")
st.write("Upload an image of a fruit and the model will tell whether it's an apple or a mango.")

# === Replace with your Google Drive MODEL FILE ID ===
# Example link: https://drive.google.com/file/d/1AbCDEFgHij12345/view?usp=sharing
# Then FILE_ID = "1AbCDEFgHij12345"
FILE_ID = "1upR2UNZEaBl4CuoW4WmyYDmxESK178Q9"
MODEL_FILE = "apple_mango_model.h5"

@st.cache_resource
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_FILE):
        st.info("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)
    model = tf.keras.models.load_model(MODEL_FILE)
    return model


model = load_model()

IMG_SIZE = (160, 160)

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Classifying..."):
        x = preprocess_image(image)
        prob = model.predict(x)[0][0]
        label_index = 1 if prob > 0.5 else 0
        class_names = ['apple', 'mango']
        st.write(f"**Prediction:** {class_names[label_index]}")
        st.write(f"**Confidence:** {prob:.3f}" if label_index==1 else f"**Confidence:** {1-prob:.3f}")
