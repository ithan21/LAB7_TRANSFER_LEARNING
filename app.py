import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import pandas as pd

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Horse vs Human Classifier",
    page_icon="🐴🧍",
    layout="centered"
)

st.title("🐴 vs 🧍 Horse vs Human Classifier")
st.write("Upload an image and the AI will predict if it's a **Horse** or a **Human**.")

# ============================================================
# LOAD THE BEST MODEL (MobileNetV2 base sa Colab results mo)
# ============================================================
MODEL_PATH = os.path.join("saved_models", "mobilenetv2_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.info("Make sure the `saved_models` folder has the `.keras` files inside.")
    st.stop()

# ============================================================
# IMAGE UPLOAD AND PREDICTION
# ============================================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image (same as training)
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict
    with st.spinner("Predicting..."):
        prediction = model.predict(img_array)
        confidence = prediction[0][0]
    
    # Display result
    st.markdown("---")
    st.subheader("Prediction Result")
    
    # Sigmoid output: < 0.5 = horse (0), >= 0.5 = human (1)
    if confidence >= 0.5:
        st.markdown(f"### 🧍 It's a **Human**!")
        st.progress(min(float(confidence), 1.0))
        st.write(f"**Confidence:** {confidence:.2%}")
    else:
        st.markdown(f"### 🐴 It's a **Horse**!")
        st.progress(min(float(1 - confidence), 1.0))
        st.write(f"**Confidence:** {(1 - confidence):.2%}")
    
    # Show raw prediction value
    st.caption(f"Raw sigmoid output: {confidence:.4f}")

# ============================================================
# COMPARISON TABLE
# ============================================================
st.markdown("---")
st.subheader("📊 Model Comparison")

csv_path = os.path.join("results", "comparison.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Highlight the best model
    best_model = df.loc[df["Test Accuracy"].idxmax(), "Model"]
    st.success(f"🏆 Best performing model: **{best_model}** (Used for this app)")
else:
    st.info("Run the training first to generate comparison results.")