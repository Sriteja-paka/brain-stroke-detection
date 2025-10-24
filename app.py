import streamlit as st
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

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

def preprocess_image(image):
    """Preprocess uploaded image using PIL"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(IMG_SIZE)
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype("float32") / 255.0
    
    return img_array

def extract_cnn_features(img_array):
    """Extract CNN features similar to training pipeline"""
    # This is a simplified version - in production you'd use the actual feature extractor
    # For now, we'll use the same preprocessing as during training
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 1:
        img_array = np.concatenate([img_array]*3, axis=2)
    
    # Apply CLAHE for contrast enhancement (similar to training)
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply((img_gray * 255).astype(np.uint8))
    img_enhanced = img_enhanced.astype(np.float32) / 255.0
    
    # Resize to match training
    img_resized = cv2.resize(img_enhanced, IMG_SIZE)
    img_final = np.stack((img_resized,)*3, axis=-1)
    
    return img_final

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
    # Read and display image using PIL
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded CT Scan", use_container_width=True)  # FIXED: use_container_width
    
    with col2:
        if st.button("🔍 Analyze for Stroke", type="primary", use_container_width=True):
            with st.spinner("Analyzing CT scan..."):
                try:
                    # Preprocess
                    img_processed = preprocess_image(image)
                    
                    # Extract features (same as training pipeline)
                    img_features = extract_cnn_features(img_processed)
                    
                    # Flatten and select features
                    feat_flat = img_features.reshape(1, -1)
                    
                    # Check feature dimensions
                    expected_features = scaler.n_features_in_
                    current_features = feat_flat.shape[1]
                    
                    st.write(f"🔍 Debug: Current features: {current_features}, Expected: {expected_features}")
                    
                    if current_features != expected_features:
                        st.error(f"Feature dimension mismatch! Got {current_features}, expected {expected_features}")
                        # Try to fix by resizing or padding
                        if current_features > expected_features:
                            feat_flat = feat_flat[:, :expected_features]
                        else:
                            feat_flat = np.pad(feat_flat, ((0, 0), (0, expected_features - current_features)), mode='constant')
                    
                    # Scale features
                    feat_scaled = scaler.transform(feat_flat)
                    
                    # Apply feature selection mask
                    feat_sel = feat_scaled[:, mask == 1]
                    
                    # Reshape for LSTM
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
                    st.info("💡 This might be due to image format or preprocessing issues. Try a different image.")

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