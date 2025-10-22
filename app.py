import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="Apple vs Mango Classifier", layout="centered")

st.title("🍎 Apple vs 🥭 Mango — Image Classifier")
st.write("Upload an image of a fruit and the model will tell whether it's an apple or a mango.")

# Path to model - for deployment, place model in same repo or load from a path
MODEL_PATH = "model_saved"  # change to the folder name you will use in deployment

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Helper: preprocess uploaded image
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
        # NOTE: change class names depending on model.class_names / training folder order
        # We assume training class order is: ['apple_train','mango_train']
        class_names = ['apple', 'mango']
        st.write(f"**Prediction:** {class_names[label_index]}")
        st.write(f"**Confidence:** {prob:.3f}" if label_index==1 else f"**Confidence:** {1-prob:.3f}")
