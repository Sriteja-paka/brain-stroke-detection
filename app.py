import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# Set page config
st.set_page_config(
    page_title="Brain Stroke Detection",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache resources
@st.cache_resource
def load_artifacts():
    """Load model and artifacts once"""
    try:
        model = load_model('best_bilstm.keras', compile=False)
        scaler = joblib.load('scaler_features.pkl')
        mask = np.load('best_mask.npy')
        return model, scaler, mask
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load artifacts
model, scaler, mask = load_artifacts()

if model is None:
    st.error("❌ Failed to load AI model. Please check if model files are available.")
    st.stop()

# Constants
IMG_SIZE = (224, 224)
SEQ_LEN = 16

def preprocess_image(img):
    """Preprocess uploaded image"""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return img

def reshape_for_lstm(Xs, seq_len=SEQ_LEN):
    """Reshape features for LSTM"""
    n, d = Xs.shape
    step = int(np.ceil(d / seq_len))
    total = step * seq_len
    if total > d:
        Xs = np.pad(Xs, ((0, 0), (0, total - d)), mode='constant')
    else:
        Xs = Xs[:, :total]
    return Xs.reshape(1, seq_len, step)

# Main UI
st.title("🧠 Brain Stroke Detection System")
st.markdown("Upload a CT scan image to detect potential stroke conditions")

# File upload
uploaded_file = st.file_uploader(
    "Choose CT scan image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear CT scan image for analysis"
)

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded CT Scan", use_column_width=True)
    
    with col2:
        if st.button("🔍 Analyze for Stroke", type="primary", use_container_width=True):
            with st.spinner("Analyzing CT scan..."):
                try:
                    # Preprocess
                    img_processed = preprocess_image(img)
                    feat_flat = img_processed.reshape(1, -1)
                    feat_scaled = scaler.transform(feat_flat)
                    feat_sel = feat_scaled[:, mask == 1]
                    seq_input = reshape_for_lstm(feat_sel)
                    
                    # Predict
                    prediction = model.predict(seq_input, verbose=0)
                    class_idx = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction)
                    
                    # Display results
                    if class_idx == 1:
                        st.error(f"🟥 **Stroke Detected**")
                        st.warning(f"Confidence: {confidence*100:.2f}%")
                        st.info("⚠️ Please consult a healthcare professional immediately!")
                    else:
                        st.success(f"🟩 **Normal Brain**")
                        st.success(f"Confidence: {confidence*100:.2f}%")
                        st.info("✅ No signs of stroke detected")
                    
                    # Confidence meter
                    st.progress(float(confidence))
                    st.caption(f"Model confidence: {confidence*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.markdown("""
    1. **Upload** a clear CT scan image (JPG, JPEG, or PNG format)
    2. **Click** the 'Analyze for Stroke' button
    3. **View** the results with confidence percentage
    4. **Important**: This is for educational purposes only
    5. **Always consult** healthcare professionals for medical diagnosis
    """)

# Footer
st.markdown("---")
st.caption("AI-Powered Stroke Detection System | For educational and research purposes")