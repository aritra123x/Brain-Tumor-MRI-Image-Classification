import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# App title
st.set_page_config(page_title="Brain Tumor Classification", layout="centered")
st.title("🧠 Brain Tumor Classification App")
st.markdown("Upload an MRI brain scan and select a model to predict the tumor type.")

# Load models with caching
@st.cache_resource
def load_all_models():
    return {
        "Custom CNN": load_model("C:/Users/KIIT/Downloads/MRI/cnn_model.h5"),
        "ResNet50": load_model("C:/Users/KIIT/Downloads/MRI/resnet_model.h5"),
        "MobileNetV2": load_model("C:/Users/KIIT/Downloads/MRI/mobilenet_model.h5"),
        "InceptionV3": load_model("C:/Users/KIIT/Downloads/MRI/inception_model.h5")
    }

# Load models
try:
    models = load_all_models()
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# Define class labels
class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing
def preprocess_image(image, target_size=(224, 224), model_name="Custom CNN"):
    image = image.resize(target_size)
    img_array = np.array(image)

    if model_name == "MobileNetV2":
        img_array = mobilenet_preprocess(img_array)
    elif model_name == "InceptionV3":
        img_array = inception_preprocess(img_array)
    elif model_name == "ResNet50":
        img_array = resnet_preprocess(img_array)
    else:
        img_array = img_array / 255.0  # Custom CNN

    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# File uploader
uploaded_file = st.file_uploader("📤 Upload an MRI image", type=['jpg', 'jpeg', 'png'])

# Model selection
selected_model_name = st.selectbox("🔍 Choose a model for prediction", list(models.keys()))

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Classifying..."):
        processed_img = preprocess_image(image, model_name=selected_model_name)
        model = models[selected_model_name]
        prediction = model.predict(processed_img)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Show result
    st.success(f"✅ **Predicted Tumor Type:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

    # Show full probabilities
    st.subheader("🔬 Prediction Probabilities:")
    for i, label in enumerate(class_labels):
        st.write(f"- **{label}**: {prediction[0][i] * 100:.2f}%")
