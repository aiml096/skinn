import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError

# Load model once
@st.cache_resource
def load_trained_model():
    return load_model("skindisease.h5")

model = load_trained_model()

classes = ['Acne', 'Melanoma', 'Peeling skin', 'Ring worm', 'Vitiligo']

st.title("🧴 Skin Disease Detection")
st.write("Upload a skin image to classify the disease.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Resize and preprocess image correctly
        img = img.resize((64, 64))
        x = img_to_array(img)  # Shape: (64, 64, 3)
        x = np.expand_dims(x, axis=0)  # Shape: (1, 64, 64, 3)

        # Check the shape before prediction
        st.write(f"Processed image shape: {x.shape}")  # For debugging

        # Predict
        preds = model.predict(x)
        label = np.argmax(preds, axis=1)[0]
        prediction = classes[label]

        st.success(f"🩺 Predicted Skin Disease: **{prediction}**")
    except UnidentifiedImageError:
        st.error("❌ Could not process the image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
