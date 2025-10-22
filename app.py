import streamlit as st
import tensorflow as tf
import gdown
import os
import zipfile

@st.cache_resource
def load_model():
    FOLDER_ID = "1EmbUNvE7Y7IwqPFwrqEyFqdlvpJ7bo-i"

    ZIP_FILE = "model.zip"
    MODEL_DIR = "apple_mango_model"

    # Download zipped model if not already present
    if not os.path.exists(MODEL_DIR):
        st.write("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={FOLDER_ID}"
        gdown.download(url, ZIP_FILE, quiet=False)

        # Unzip the model
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    # Load model from folder
    model = tf.keras.models.load_model(MODEL_DIR)
    return model
