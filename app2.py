import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, UnidentifiedImageError

# Load the model (use st.cache_resource to avoid reloading every time)
@st.cache_resource
def load_trained_model():
    return load_model("skindisease.h5")

model = load_trained_model()

# Class labels
classes = ['Acne', 'Melanoma', 'Peeling skin', 'Ring worm', 'Vitiligo']

# App UI
st.title("🧴 Skin Disease Detection")
st.write("Upload an image of a skin condition, and the model will predict the disease.")

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Open and display image
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = img.resize((64, 64))  # Ensure size matches training input
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension

        # Predict
        preds = model.predict(x)
        st.write("Raw model output:", preds)  # Debug output
        label = np.argmax(preds, axis=1)[0]
        prediction = classes[label]

        st.success(f"🩺 Predicted Skin Disease: **{prediction}**")

    except UnidentifiedImageError:
        st.error("❌ Unable to open image. Please upload a valid image file.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {e}")
